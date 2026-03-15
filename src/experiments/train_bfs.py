# This script handles the training loop for the BFS task.

import os
import sys

# Prevent TensorFlow (imported transitively by dm-clrs) from pre-allocating all GPU memory
os.environ["CUDA_VISIBLE_DEVICES_TF"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# Reduce CUDA fragmentation for variable-size graph batches
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import csv
import glob
import time
import torch
from loguru import logger
import lightning.pytorch as pl
import argparse
import wandb # Added import

# Add project root to sys.path so absolute imports work when running directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.models.module import SALSACLRSModel
from src.utils.config import load_cfg
from src.utils.utils import NaNException
from src.utils.graph_generation import get_dataset

from salsaclrs import SALSACLRSDataModule

logger.remove()
logger.add(sys.stderr, level="INFO")

os.environ["OMP_NUM_THREADS"] = "4" 
torch.set_num_threads(4)

class EpochProfilingCallback(pl.Callback):
    def __init__(self, every_n_epochs=100):
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.start_time = 0.0
        self.epoch_times = []
        self.batch_counts = []

    def on_train_epoch_start(self, trainer, pl_module):
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        epoch_time = time.time() - self.start_time
        self.epoch_times.append(epoch_time)
        self.batch_counts.append(trainer.num_training_batches)

        if (trainer.current_epoch + 1) % self.every_n_epochs == 0:
            avg_time = sum(self.epoch_times) / len(self.epoch_times)
            total_time = sum(self.epoch_times)
            total_batches = sum(self.batch_counts)
            it_s = total_batches / total_time if total_time > 0 else 0
            
            # Log to stdout via loguru
            logger.info(f"Epoch {trainer.current_epoch + 1} Profiling: Avg time/epoch: {avg_time:.2f}s | it/s: {it_s:.2f}")
            
            # Log to wandb if available
            if trainer.logger and isinstance(trainer.logger, pl.loggers.WandbLogger):
                trainer.logger.experiment.log({
                    "profiling/time_per_epoch": avg_time,
                    "profiling/it_per_s": it_s,
                    "epoch": trainer.current_epoch
                })
                
            self.epoch_times.clear()
            self.batch_counts.clear()

def train(model, datamodule, cfg, specs, seed=42, checkpoint_dir=None, enable_wandb=False, enable_progress_bar=False, fast_dev_run=False):
    # Enable TF32 for matrix multiplications (massive speedup on Ampere/Ada/Blackwell GPUs)
    torch.set_float32_matmul_precision('high')
    if enable_wandb:
        wandblogger = pl.loggers.WandbLogger(project=cfg.LOGGING.WANDB.PROJECT, group=cfg.LOGGING.WANDB.GROUP, name=cfg.RUN_NAME+"-"+str(seed), log_model="all")
    else:
        wandblogger = None


    # Determine the monitor metric dynamically
    val_nickname = datamodule.get_val_loader_nickname(0)
    monitor_metric = cfg.TRAIN.CHECKPOINT_MONITOR.format(val_nickname=val_nickname)
    monitor_mode = cfg.TRAIN.CHECKPOINT_MONITOR_MODE

    ckpt_dir_path = os.path.join(cfg.DATA.ROOT, "checkpoints", str(cfg.ALGORITHM), cfg.RUN_NAME) if checkpoint_dir is not None else None

    def _make_callbacks():
        """Create fresh callback instances (required for each new Trainer)."""
        cbs = []
        _ckpt_cbk = None
        if checkpoint_dir is not None:
            _ckpt_cbk = pl.callbacks.ModelCheckpoint(dirpath=ckpt_dir_path, monitor=monitor_metric, mode=monitor_mode, filename=f'seed{seed}-{{epoch}}-{{step}}', save_top_k=1, save_last=False)
            cbs.append(_ckpt_cbk)
            cbs.append(pl.callbacks.ModelCheckpoint(
                dirpath=ckpt_dir_path,
                filename=f'seed{seed}-periodic-{{epoch:03d}}',
                every_n_epochs=250,
                save_top_k=-1,
                save_last=False,
            ))
        cbs.append(pl.callbacks.EarlyStopping(monitor=monitor_metric, patience=cfg.TRAIN.EARLY_STOPPING_PATIENCE, mode=monitor_mode, check_finite=False))
        cbs.append(EpochProfilingCallback(every_n_epochs=100))
        if enable_progress_bar:
            from lightning.pytorch.callbacks import TQDMProgressBar
            cbs.append(TQDMProgressBar(refresh_rate=20))
        return cbs, _ckpt_cbk

    def _make_trainer(logger_inst):
        """Create a fresh Trainer instance."""
        cbs, _ckpt_cbk = _make_callbacks()
        _trainer = pl.Trainer(
            enable_checkpointing=checkpoint_dir is not None,
            callbacks=cbs,
            max_epochs=cfg.TRAIN.MAX_EPOCHS,
            logger=logger_inst,
            accelerator="auto",
            log_every_n_steps=3,
            check_val_every_n_epoch=50,
            gradient_clip_val=cfg.TRAIN.GRADIENT_CLIP_VAL,
            accumulate_grad_batches=cfg.TRAIN.GRADIENT_ACCUMULATION_STEPS,
            fast_dev_run=fast_dev_run,
            reload_dataloaders_every_n_epochs=datamodule.reload_every_n_epochs,
            precision=cfg.TRAIN.PRECISION,
            enable_progress_bar=enable_progress_bar,
        )
        return _trainer, _ckpt_cbk

    trainer, ckpt_cbk = _make_trainer(wandblogger)

    # Load checkpoint
    resume_path = None
    if cfg.TRAIN.LOAD_CHECKPOINT is not None:
        logger.info(f"Loading checkpoint from {cfg.TRAIN.LOAD_CHECKPOINT}")
        # Note: If we just want to resume training, pl.Trainer.fit(..., ckpt_path=...) handles it.
        # But if we also want testing without training enabled, load it into model directly:
        model = SALSACLRSModel.load_from_checkpoint(cfg.TRAIN.LOAD_CHECKPOINT, cfg=cfg, specs=specs)
        resume_path = cfg.TRAIN.LOAD_CHECKPOINT

    # Train
    if cfg.TRAIN.ENABLE:
        logger.info("PRE-COLLATING DATASET INTO VRAM/RAM...")
        import random
        import torch.multiprocessing as mp
        mp.set_sharing_strategy('file_system')
        
        orig_dl = datamodule.train_dataloader()
        static_batches = []
        
        # Iterate through native PyG collater EXACTLY ONCE
        for batch in orig_dl:
            static_batches.append(batch.clone())
            
        logger.info(f"Created {len(static_batches)} static Super-Batches.")

        # 4. Create a dummy DataLoader that just yields our pre-made batches
        class StaticLoader:
            def __iter__(self):
                # Shuffle the static batches to maintain SGD noise across epochs
                random.shuffle(static_batches)
                return iter(static_batches)
            def __len__(self):
                return len(static_batches)

        # 5. Overwrite the datamodule's train loader
        datamodule.train_dataloader = lambda: StaticLoader()

        try:
            try:
                logger.info("Starting training...")
                trainer.fit(model, datamodule=datamodule, ckpt_path=resume_path)
                logger.info(f"Training finished at epoch {trainer.current_epoch}/{cfg.TRAIN.MAX_EPOCHS}")
            except NaNException:
                # Find the most recent periodic checkpoint; fall back to best-metric checkpoint
                recover_path = None
                if ckpt_dir_path:
                    periodic_files = glob.glob(os.path.join(ckpt_dir_path, f'seed{seed}-periodic-*.ckpt'))
                    if periodic_files:
                        recover_path = max(periodic_files, key=os.path.getmtime)
                if recover_path is None and ckpt_cbk and ckpt_cbk.best_model_path:
                    recover_path = ckpt_cbk.best_model_path
                
                if recover_path and not os.path.exists(recover_path):
                    logger.warning(f"Fallback checkpoint {recover_path} not found.")
                    if ckpt_dir_path:
                        all_ckpts = glob.glob(os.path.join(ckpt_dir_path, '*.ckpt'))
                        valid_ckpts = [ckpt for ckpt in all_ckpts if 'final' not in ckpt]
                        if valid_ckpts:
                            recover_path = max(valid_ckpts, key=os.path.getmtime)
                            logger.info(f"Found latest checkpoint manually: {recover_path}")
                        else:
                            recover_path = None
                    else:
                        recover_path = None

                if recover_path:
                    logger.info(f"NaN detected, recovering from {recover_path} with a fresh Trainer...")
                    # A Trainer cannot be reused after an exception — its internal
                    # state machine is corrupted and will fast-forward through epochs.
                    trainer, ckpt_cbk = _make_trainer(wandblogger)
                    try:
                        trainer.fit(model, datamodule=datamodule, ckpt_path=recover_path)
                        logger.info(f"Recovery training finished at epoch {trainer.current_epoch}/{cfg.TRAIN.MAX_EPOCHS}")
                    except NaNException:
                        logger.info("Recovery failed, stopping training...")
                else:
                    logger.info("NaN detected but no checkpoint available for recovery.")
        except KeyboardInterrupt:
            logger.info("Training interrupted via KeyboardInterrupt. Proceeding to save final checkpoint...")

    # Save final checkpoint explicitly
    if checkpoint_dir is not None and cfg.TRAIN.ENABLE:
        final_ckpt_path = os.path.join(checkpoint_dir, str(cfg.ALGORITHM), cfg.RUN_NAME, f'seed{seed}-final.ckpt')
        trainer.save_checkpoint(final_ckpt_path)
        logger.info(f"Saved final checkpoint to {final_ckpt_path}")
        
        if enable_wandb and wandblogger is not None:
            artifact = wandb.Artifact(name=f"model-{wandblogger.experiment.id}-final", type="model")
            artifact.add_file(final_ckpt_path)
            wandblogger.experiment.log_artifact(artifact)

    # Load best model
    # if cfg.TRAIN.LOAD_CHECKPOINT is None and cfg.TRAIN.ENABLE:
    #     if ckpt_cbk and ckpt_cbk.best_model_path:
    #         logger.info(f"Best model path: {ckpt_cbk.best_model_path}")
    #         model = SALSACLRSModel.load_from_checkpoint(ckpt_cbk.best_model_path)
    #     else:
    #         logger.warning("No best model found. Testing with current weights.")

    # # Test
    # logger.info("Testing best model...")
    logger.info("Testing final model...")
    results = trainer.test(model, datamodule=datamodule)

    # Log results
    stacked_results = {}
    for d in results:
        stacked_results.update(d)

    logger.info(stacked_results)
    logger.info("Saving results...")
    results_dir = f"results/{cfg.ALGORITHM}/{cfg.RUN_NAME}"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)

    # write csv
    with open(os.path.join(results_dir, f"{seed}.csv"), "w") as f:
        writer = csv.DictWriter(f, stacked_results.keys())
        writer.writeheader()
        writer.writerow(stacked_results)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True, help="Path to config file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--hints", action="store_true", help="Use hints.")
    parser.add_argument("--enable-wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--enable-progress-bar", action="store_true", help="Enable tqdm progress bars")
    parser.add_argument("--fast_dev_run", action="store_true", help="Run 1 train, val and test loop")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER, help="Modify config options from command line")
    args = parser.parse_args()

    # set seed
    pl.seed_everything(args.seed)
    logger.info(f"Using seed {args.seed}")

    # load config
    cfg = load_cfg(args.cfg, args.opts)

    DATA_DIR = cfg.DATA.ROOT

    if args.hints:
        cfg.TRAIN.LOSS.HINT_LOSS_WEIGHT = 1.0
        cfg.RUN_NAME = cfg.RUN_NAME+"-hints"
        logger.info("Using hints.")

    
    logger.info("Starting run...")
    torch.set_float32_matmul_precision('medium')

    # load datasets
    train_ds = get_dataset("train",cfg)
    val_ds = get_dataset("val",cfg)
    test_datasets = get_dataset("test",cfg)
    specs = train_ds.specs
    
    # load model
    datamodule = SALSACLRSDataModule(
        train_dataset=train_ds,
        val_datasets=list(val_ds.values()), 
        test_datasets=list(test_datasets.values()), 
        batch_size=cfg.TRAIN.BATCH_SIZE, 
        num_workers=cfg.TRAIN.NUM_WORKERS, 
        test_batch_size=cfg.TEST.BATCH_SIZE
    )

    # Monkeypatch: force persistent_workers=False (required when num_workers=0, e.g. on Kaggle)
    # Also enforce pin_memory and persistent_workers when num_workers > 0
    _orig_dataloader = datamodule.dataloader
    def _patched_dataloader(dataset, **kwargs):
        # salsaclrs overrides num_workers to 0 during testing
        current_num_workers = kwargs.get("num_workers", cfg.TRAIN.NUM_WORKERS)
        if current_num_workers == 0:
            kwargs["persistent_workers"] = False
            kwargs["pin_memory"] = False
        else:
            kwargs["persistent_workers"] = True
            kwargs["pin_memory"] = True
        return _orig_dataloader(dataset, **kwargs)
    datamodule.dataloader = _patched_dataloader
    datamodule.val_dataloader()
    model = SALSACLRSModel(specs=train_ds.specs, cfg=cfg)

    ckpt_dir = os.path.join(DATA_DIR, "checkpoints")
    train(model, datamodule, cfg, train_ds.specs, seed = args.seed, checkpoint_dir=ckpt_dir, enable_wandb=args.enable_wandb, enable_progress_bar=args.enable_progress_bar, fast_dev_run=args.fast_dev_run)

    if args.enable_wandb:
        wandb.finish()