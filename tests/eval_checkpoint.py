"""
Script to evaluate a trained model checkpoint on the test set.

Usage:
    python tests/eval_checkpoint.py --cfg src/configs/bfs/PGN-grokking.yml --ckpt path/to/model.ckpt
"""

import os
import sys
import argparse
import torch
import lightning.pytorch as pl

# Add project root to sys.path so absolute imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.module import SALSACLRSModel
from src.utils.graph_generation import get_dataset
from salsaclrs import SALSACLRSDataModule
from loguru import logger

def format_results_table(results):
    """
    Format the raw PyTorch Lightning test results into a nice markdown-style table.
    """
    if not results:
        return "No results to display."

    # results is a list of dicts (one per dataloader)
    # Let's merge them into one big dict
    merged_results = {}
    for d in results:
        merged_results.update(d)

    # Group metrics by dataset
    # Metrics are usually named like: "test/graph_accuracy/ws_800"
    datasets_metrics = {}
    metric_names = set()

    for key, value in merged_results.items():
        if not key.startswith("test/"):
            continue
            
        parts = key.split("/")
        if len(parts) >= 3:
            metric_type = parts[1]
            dataset_name = "/".join(parts[2:])
            
            if dataset_name not in datasets_metrics:
                datasets_metrics[dataset_name] = {}
                
            datasets_metrics[dataset_name][metric_type] = value
            metric_names.add(metric_type)

    if not datasets_metrics:
        # Fallback if names don't match the pattern
        return "\n".join(f"{k}: {v:.4f}" for k, v in merged_results.items())

    # Sort names for consistent columns
    metric_names = sorted(list(metric_names))
    dataset_names = sorted(list(datasets_metrics.keys()))

    # Build the table
    col_width = max(len(name) for name in dataset_names) + 2
    col_width = max(col_width, 15)
    
    # Header
    header = f"| {'Dataset'.ljust(col_width)} |"
    for metric in metric_names:
        header += f" {metric.ljust(15)} |"
    
    separator = "|" + "-" * (col_width + 2) + "|"
    for _ in metric_names:
        separator += "-" * 17 + "|"

    output = []
    output.append(header)
    output.append(separator)

    # Rows
    for dataset in dataset_names:
        row = f"| {dataset.ljust(col_width)} |"
        for metric in metric_names:
            val = datasets_metrics[dataset].get(metric, 0.0)
            if isinstance(val, torch.Tensor):
                val = val.item()
            row += f" {val:<15.4f} |" if isinstance(val, (int, float)) else f" {str(val).ljust(15)} |"
        output.append(row)

    return "\n".join(output)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of dataloader workers")
    args = parser.parse_args()

    pl.seed_everything(args.seed)
    
    # Configure loguru to suppress excessive debug prints
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    # Ensure precision is set right
    torch.set_float32_matmul_precision('medium')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load Model (this naturally restores its saved hparams, like cfg and specs)
    logger.info(f"Loading checkpoint {args.ckpt} with strict=False for backward compatibility")
    model = SALSACLRSModel.load_from_checkpoint(args.ckpt, map_location=device, strict=False)
    
    # Get cfg directly from the loaded model
    cfg = model.cfg

    # Add 1600-node graphs to TEST manually if they are not already there
    if "er_1600" not in cfg.DATA.TEST.NICKNAME:
        cfg.DATA.TEST.GRAPH_GENERATOR.append("er")
        cfg.DATA.TEST.NICKNAME.append("er_1600")
        cfg.DATA.TEST.GENERATOR_PARAMS.append({"n": 1600, "p_range": [0.004, 0.0125], "directed": False, "acyclic": False, "weighted": False})
        
    if "ws_1600" not in cfg.DATA.TEST.NICKNAME:
        cfg.DATA.TEST.GRAPH_GENERATOR.append("ws")
        cfg.DATA.TEST.NICKNAME.append("ws_1600")
        cfg.DATA.TEST.GENERATOR_PARAMS.append({"n": 1600, "p_range": [0.05, 0.2], "k": [4, 6, 8], "directed": False, "acyclic": False, "weighted": False})
        
    if "delaunay_1600" not in cfg.DATA.TEST.NICKNAME:
        cfg.DATA.TEST.GRAPH_GENERATOR.append("delaunay")
        cfg.DATA.TEST.NICKNAME.append("delaunay_1600")
        cfg.DATA.TEST.GENERATOR_PARAMS.append({"n": 1600, "directed": False, "acyclic": False, "weighted": False})

    # Load Data
    logger.info("Loading test datasets...")
    test_datasets_dict = get_dataset("test", cfg)
    
    datamodule = SALSACLRSDataModule(
        train_dataset=None,  # Not needed for testing
        val_datasets=[],     # Not needed for testing
        test_datasets=list(test_datasets_dict.values()), 
        batch_size=cfg.TRAIN.BATCH_SIZE, 
        num_workers=args.num_workers, 
        test_batch_size=cfg.TEST.BATCH_SIZE
    )

    # Monkeypatch for Kaggle/zero workers issues
    if args.num_workers == 0:
        _orig_dataloader = datamodule.dataloader
        def _patched_dataloader(dataset, **kwargs):
            kwargs["persistent_workers"] = False
            return _orig_dataloader(dataset, **kwargs)
        datamodule.dataloader = _patched_dataloader

    # Init Trainer
    trainer = pl.Trainer(
        accelerator="auto",
        logger=False, # Disable wandb logging for pure eval
        precision=cfg.TRAIN.PRECISION,
    )

    # Run Eval
    logger.info("Running evaluation...")
    results = trainer.test(model, datamodule=datamodule)

    # Print Table
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    table = format_results_table(results)
    print(table)
    print("="*80 + "\n")

if __name__ == '__main__':
    main()
