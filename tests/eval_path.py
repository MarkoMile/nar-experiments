"""
Script to evaluate a trained model checkpoint on path graphs with controllable
maximum BFS depth.

Path graphs are worst-case for BFS depth: a path of n nodes has depth n-1
when the source is at an endpoint. By varying n we directly control the
maximum BFS depth the model must handle.

Usage:
    # Evaluate on default path sizes (depths 16, 32, 64, 128, 256, 512)
    python tests/eval_path.py --ckpt path/to/model.ckpt

    # Custom max depths
    python tests/eval_path.py --ckpt path/to/model.ckpt --max-depths 8 16 32 64 128

    # More samples for statistical significance
    python tests/eval_path.py --ckpt path/to/model.ckpt --num-samples 50
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
    # Metrics are usually named like: "test/graph_accuracy/path_32"
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

    import re
    def natural_sort_key(s):
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

    # Sort names for consistent columns
    metric_names = sorted(list(metric_names))
    dataset_names = sorted(list(datasets_metrics.keys()), key=natural_sort_key)

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
    parser = argparse.ArgumentParser(
        description="Evaluate a BFS model on path graphs with controllable max depth."
    )
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of dataloader workers")
    parser.add_argument("--num-samples", type=int, default=15, help="Number of samples per path size")
    parser.add_argument(
        "--max-depths", type=int, nargs="+",
        default=[16, 32, 64, 128, 256, 512],
        help="Max BFS depths to test. Each depth d creates a path graph with n=d+1 nodes. "
             "Default: 16 32 64 128 256 512"
    )
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

    # Build the test dataset configuration from --max-depths
    # A path graph with n nodes has max BFS depth = n-1 (source at endpoint).
    # With a random source the actual depth is max(s, n-1-s) where s is the source index.
    generators = []
    nicknames = []
    generator_params = []

    for depth in sorted(args.max_depths):
        n_nodes = depth + 1  # path of n nodes → max depth n-1
        generators.append("path")
        nicknames.append(f"path_d{depth}")
        generator_params.append({"n": n_nodes})

    cfg.DATA.TEST.NUM_SAMPLES = args.num_samples
    cfg.DATA.TEST.GRAPH_GENERATOR = generators
    cfg.DATA.TEST.NICKNAME = nicknames
    cfg.DATA.TEST.GENERATOR_PARAMS = generator_params

    # Print test plan
    logger.info(f"Evaluating on {len(args.max_depths)} path graph configurations:")
    for nick, params in zip(nicknames, generator_params):
        logger.info(f"  {nick}: n={params['n']} nodes (max BFS depth = {params['n']-1})")

    # Load Data
    logger.info(f"Loading path graph test datasets (Samples per set: {args.num_samples})...")
    test_datasets_dict = get_dataset("test", cfg, seed=args.seed)
    
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
    print("EVALUATION RESULTS (Path Graphs — Max Depth Sweep)")
    print("="*80)
    table = format_results_table(results)
    print(table)
    print("="*80 + "\n")

if __name__ == '__main__':
    main()
