# Training script for the Bellman-Ford task.
# Uses the same algorithm-agnostic pipeline as BFS — only the config differs.
#
# Usage:
#   python -m src.experiments.train_bellman_ford --cfg src/configs/bellman_ford/bestmodel.yml --enable-wandb

import os
import sys

# Prevent TensorFlow (imported transitively by dm-clrs) from pre-allocating all GPU memory
os.environ["CUDA_VISIBLE_DEVICES_TF"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# Reduce CUDA fragmentation for variable-size graph batches
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from loguru import logger
import lightning.pytorch as pl
import argparse
import wandb

# Add project root to sys.path so absolute imports work when running directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.models.module import SALSACLRSModel
from src.utils.config import load_cfg
from src.utils.graph_generation import get_dataset

from salsaclrs import SALSACLRSDataModule

# Reuse the training function and callback from train_bfs
from src.experiments.train_bfs import train, EpochProfilingCallback

logger.remove()
logger.add(sys.stderr, level="INFO")

os.environ["OMP_NUM_THREADS"] = "4"
torch.set_num_threads(4)


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

    
    logger.info(f"Starting Bellman-Ford run: {cfg.RUN_NAME}")
    torch.set_float32_matmul_precision('medium')

    # load datasets
    train_ds = get_dataset("train", cfg, seed=args.seed)
    val_ds = get_dataset("val", cfg, seed=args.seed)
    test_datasets = get_dataset("test", cfg, seed=args.seed)
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
    train(model, datamodule, cfg, train_ds.specs, seed=args.seed, checkpoint_dir=ckpt_dir, enable_wandb=args.enable_wandb, enable_progress_bar=args.enable_progress_bar, fast_dev_run=args.fast_dev_run)

    if args.enable_wandb:
        wandb.finish()
