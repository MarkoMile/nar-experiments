# This script handles the training loop for the BFS task.

import os
import sys
import csv
import torch
from loguru import logger
import lightning.pytorch as pl
import argparse

# Add project root to sys.path so absolute imports work when running directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.models.module import SALSACLRSModel
from src.utils.config import load_cfg
from src.utils.utils import NaNException
from src.utils.graph_generation import get_dataset

from salsaclrs import SALSACLRSDataModule

logger.remove()
logger.add(sys.stderr, level="INFO")

def train(model, datamodule, cfg, specs, seed=42, checkpoint_dir=None):
    callbacks = []
    # checkpointing
    if checkpoint_dir is not None:
        ckpt_cbk = pl.callbacks.ModelCheckpoint(dirpath=os.path.join(cfg.DATA.ROOT, "checkpoints", str(cfg.ALGORITHM), cfg.RUN_NAME), monitor="val/outloss/0", mode="min", filename=f'seed{seed}-{{epoch}}-{{step}}', save_top_k=1, save_last=True)
        callbacks.append(ckpt_cbk)
    else:
        ckpt_cbk = None

    # early stopping
    # early_stop_cbk = pl.callbacks.EarlyStopping(monitor="val/outloss/0", patience=cfg.TRAIN.EARLY_STOPPING_PATIENCE, mode="min")
    # callbacks.append(early_stop_cbk)

    # callbacks.append(pl.callbacks.RichProgressBar()) # <--- REMOVED FOR BETTER LOGGING IN KAGGLE
    from lightning.pytorch.callbacks import TQDMProgressBar
    callbacks.append(TQDMProgressBar(refresh_rate=20))

    trainer = pl.Trainer(
        enable_checkpointing=checkpoint_dir is not None,
        callbacks=callbacks,
        max_epochs=cfg.TRAIN.MAX_EPOCHS,
        logger=None,
        accelerator="auto",
        log_every_n_steps=5,
        gradient_clip_val=cfg.TRAIN.GRADIENT_CLIP_VAL,
        reload_dataloaders_every_n_epochs=datamodule.reload_every_n_epochs,
        precision=cfg.TRAIN.PRECISION,
        enable_progress_bar=True,
    )

    # Load checkpoint
    if cfg.TRAIN.LOAD_CHECKPOINT is not None:
        logger.info(f"Loading checkpoint from {cfg.TRAIN.LOAD_CHECKPOINT}")
        model = SALSACLRSModel.load_from_checkpoint(cfg.TRAIN.LOAD_CHECKPOINT, cfg=cfg, specs=specs)

    # Train
    if cfg.TRAIN.ENABLE:
        try:
            logger.info("Starting training...")
            trainer.fit(model, datamodule=datamodule)
        except NaNException:
            logger.info(f"NaN detected, trying to recover from {ckpt_cbk.best_model_path}...")
            try:
                trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_cbk.best_model_path)
            except NaNException:
                logger.info("Recovery failed, stopping training...")

    # Load best model
    if cfg.TRAIN.LOAD_CHECKPOINT is None and cfg.TRAIN.ENABLE:
        logger.info(f"Best model path: {ckpt_cbk.best_model_path}")
        model = SALSACLRSModel.load_from_checkpoint(ckpt_cbk.best_model_path)

    # Test
    logger.info("Testing best model...")
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
    args = parser.parse_args()

    # set seed
    pl.seed_everything(args.seed)
    logger.info(f"Using seed {args.seed}")

    # load config
    cfg = load_cfg(args.cfg)

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
    test_ds_small = test_datasets['er_80']
    test_ds_large = test_datasets['er_800']
    specs = train_ds.specs
    
    # load model
    datamodule = SALSACLRSDataModule(train_dataset=train_ds,val_datasets=[val_ds], test_datasets=[test_ds_small,test_ds_large], batch_size=cfg.TRAIN.BATCH_SIZE, num_workers=cfg.TRAIN.NUM_WORKERS, test_batch_size=cfg.TEST.BATCH_SIZE)

    # Monkeypatch: force persistent_workers=False (required when num_workers=0, e.g. on Kaggle)
    _orig_dataloader = datamodule.dataloader
    def _patched_dataloader(dataset, **kwargs):
        kwargs["persistent_workers"] = False
        return _orig_dataloader(dataset, **kwargs)
    datamodule.dataloader = _patched_dataloader

    datamodule.val_dataloader()
    model = SALSACLRSModel(specs=train_ds.specs, cfg=cfg)

    ckpt_dir = os.path.join(DATA_DIR, "checkpoints")
    train(model, datamodule, cfg, train_ds.specs, seed = args.seed, checkpoint_dir=ckpt_dir)