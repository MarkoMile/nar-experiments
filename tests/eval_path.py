"""
Script to evaluate a trained model checkpoint on path graphs with controllable
maximum BFS depth.

Path graphs are worst-case for BFS depth: a path of n nodes has depth n-1
when the source is at an endpoint. By varying n we directly control the
maximum BFS depth the model must handle.

Reports a per-depth mistake distribution showing where predictions break down.

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

# Add project root to sys.path so absolute imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.module import SALSACLRSModel
from src.utils.graph_generation import get_dataset
from salsaclrs import SALSACLRSDataModule
from loguru import logger
from tests.bfs_depth_analysis import analyse_batch, print_report


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a BFS model on path graphs with controllable max depth."
    )
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-samples", type=int, default=15, help="Number of samples per path size")
    parser.add_argument("--batch-size", type=int, default=1, help="Inference batch size")
    parser.add_argument("--device", type=str, default="auto", help="cpu / cuda / auto")
    parser.add_argument(
        "--max-depths", type=int, nargs="+",
        default=[16, 32, 64, 128, 256, 512],
        help="Max BFS depths to test. Each depth d creates a path graph with n=d+1 nodes. "
             "Default: 16 32 64 128 256 512"
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # Configure loguru to suppress excessive debug prints
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    # --- Config & device ---
    torch.set_float32_matmul_precision('medium')
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # --- Load model ---
    print(f"Loading checkpoint: {args.ckpt}")
    model = SALSACLRSModel.load_from_checkpoint(args.ckpt, map_location=device, strict=False)
    cfg = model.cfg
    model.eval()
    model.to(device)

    # --- Find output key ---
    specs = model.specs
    output_key = None
    for k, v in specs.items():
        stage = v[0]
        stage_name = stage.name if hasattr(stage, "name") else str(stage)
        if stage_name.upper() == "OUTPUT":
            output_key = k
            break
    assert output_key is not None, "Could not find OUTPUT key in specs"
    print(f"Output key: {output_key}\n")

    ignore_hints = (cfg.TRAIN.LOSS.HINT_LOSS_WEIGHT == 0.0)
    data_root = os.path.join(cfg.DATA.ROOT, "salsaclrs")
    os.makedirs(data_root, exist_ok=True)

    # --- Evaluate each path depth config ---
    from salsaclrs import SALSACLRSDataset

    for depth in sorted(args.max_depths):
        n_nodes = depth + 1  # path of n nodes → max depth n-1
        config_name = f"path_d{depth}"
        print(f"▸ {config_name}: generating {args.num_samples} samples (n={n_nodes} nodes) …")

        dataset = SALSACLRSDataset(
            root=data_root,
            split="test",
            algorithm=cfg.ALGORITHM,
            num_samples=args.num_samples,
            graph_generator="path",
            graph_generator_kwargs={"n": n_nodes},
            verify_duplicates=False,
            ignore_all_hints=ignore_hints,
            nickname=f"{config_name}_depth",
        )

        datamodule = SALSACLRSDataModule(
            test_datasets=[dataset],
            batch_size=args.batch_size,
            num_workers=0,
            test_batch_size=args.batch_size,
        )

        all_results = []
        with torch.no_grad():
            for loader in datamodule.test_dataloader():
                for batch in loader:
                    batch = batch.to(device)
                    output, hints, hidden = model(batch)
                    all_results.extend(analyse_batch(batch, output, output_key, device))

        print_report(config_name, all_results)


if __name__ == '__main__':
    main()
