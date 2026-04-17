"""
Script to evaluate a trained model checkpoint on the ogbn-arxiv dataset subgraphs.

Usage:
    python tests/test_obgn_arxiv.py --ckpt path/to/model.ckpt
"""

import os
import sys
import argparse
import torch
import numpy as np
import lightning.pytorch as pl

# Add project root to sys.path so absolute imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.module import SALSACLRSModel
from src.utils.graph_generation import get_dataset
from salsaclrs import SALSACLRSDataModule
from salsaclrs.sampler import Sampler
from loguru import logger
from tests.utils.arxiv_loader import arxiv_graph_generator

# Import the format_results_table function from eval_checkpoint
from tests.eval_checkpoint import format_results_table
from tests.bfs_depth_analysis import analyse_batch, print_report

# Monkeypatch Sampler to support our Arxiv generator
from src.utils.graph_generation import patched_create_graph as existing_patched_create

def arxiv_aware_create_graph(self, n, weighted, directed, low=0.0, high=1.0, **kwargs):
    if self._graph_generator == "arxiv":
        n_val = self._select_parameter(n)
        return arxiv_graph_generator(n_val, seed=None)
    else:
        return existing_patched_create(self, n, weighted, directed, low=low, high=high, **kwargs)

Sampler._create_graph = arxiv_aware_create_graph

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of dataloader workers")
    parser.add_argument("--num-samples", type=int, default=1, help="Samples to generate per magnitude")
    args = parser.parse_args()

    pl.seed_everything(args.seed)
    
    # Configure loguru to suppress excessive debug prints
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    # Ensure precision is set right
    torch.set_float32_matmul_precision('medium')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load Model
    logger.info(f"Loading checkpoint {args.ckpt} with strict=False for backward compatibility")
    model = SALSACLRSModel.load_from_checkpoint(args.ckpt, map_location=device, strict=False)
    
    # Get cfg directly from the loaded model
    cfg = model.cfg

    # Override TEST datasets to only run Arxiv
    cfg.DATA.TEST.NUM_SAMPLES = args.num_samples
    cfg.DATA.TEST.GRAPH_GENERATOR = ["arxiv", "arxiv", "arxiv", "arxiv", "arxiv"]
    cfg.DATA.TEST.NICKNAME = ["arxiv_16", "arxiv_80", "arxiv_800", "arxiv_1600", "arxiv_16000"]
    cfg.DATA.TEST.GENERATOR_PARAMS = [
        {"n": 16, "directed": False, "acyclic": False, "weighted": False},
        {"n": 80, "directed": False, "acyclic": False, "weighted": False},
        {"n": 800, "directed": False, "acyclic": False, "weighted": False},
        {"n": 1600, "directed": False, "acyclic": False, "weighted": False},
        {"n": 16000, "directed": False, "acyclic": False, "weighted": False}
    ]

    # Load Data
    logger.info("Loading test datasets (OGBN-Arxiv subgraphs)...")
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
    logger.info("Running standard evaluation on OGBN-Arxiv subgraphs...")
    results = trainer.test(model, datamodule=datamodule)

    # Print Table
    print("\n" + "="*80)
    print("ARXIV EVALUATION RESULTS")
    print("="*80)
    table = format_results_table(results)
    print(table)
    print("="*80 + "\n")

    # Run depth/reachability analysis similar to tests/bfs_depth_analysis.py
    logger.info("Running depth analysis and extracting statistics...")
    
    specs = model.specs
    output_key = None
    for k, v in specs.items():
        stage = v[0]
        stage_name = stage.name if hasattr(stage, "name") else str(stage)
        if stage_name.upper() == "OUTPUT":
            output_key = k
            break

    if output_key is not None:
        model.eval()
        model.to(device)
        for idx, loader in enumerate(datamodule.test_dataloader()):
            name = datamodule.get_test_loader_nickname(idx)
            all_results = []
            with torch.no_grad():
                for batch in loader:
                    batch = batch.to(device)
                    output, hints, hidden = model(batch)
                    all_results.extend(analyse_batch(batch, output, output_key, device))
            
            print_report(name, all_results)
    else:
        logger.warning("Could not find OUTPUT key in model specs; skipping depth analysis.")

if __name__ == '__main__':
    main()
