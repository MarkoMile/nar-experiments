"""
Script to evaluate a trained model checkpoint on path graphs with controllable
maximum BFS depth.

Path graphs are worst-case for BFS depth: a path of n nodes has depth n-1
when the source is at an endpoint. By varying n we directly control the
maximum BFS depth the model must handle.

Reports per-depth mistake distribution showing where predictions break down,
using the same correctness logic as the model's own calc_metrics (pointer argmax).

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
import contextlib
import torch
import numpy as np
from collections import defaultdict

# Add project root to sys.path so absolute imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.module import SALSACLRSModel, calc_metrics
from src.utils.graph_generation import get_dataset
from salsaclrs import SALSACLRSDataModule, SALSACLRSDataset
from loguru import logger


# ---------------------------------------------------------------------------
# Depth analysis helpers
# ---------------------------------------------------------------------------

def node_depths_from_predecessors(pred):
    """Return per-node depth array from a predecessor array.

    pred[i] = parent of node i.  Root has pred[root] = root.
    """
    n = len(pred)
    depths = np.full(n, -1, dtype=int)

    def _depth(i, visited):
        if depths[i] >= 0:
            return depths[i]
        if pred[i] == i:
            depths[i] = 0
            return 0
        if i in visited:          # cycle – shouldn't happen in valid BFS tree
            depths[i] = 0
            return 0
        visited.add(i)
        depths[i] = 1 + _depth(pred[i], visited)
        return depths[i]

    for i in range(n):
        if depths[i] < 0:
            _depth(i, set())

    return depths


def analyse_batch(batch, output, output_key, type_):
    """Analyse one batch using the same correctness logic as calc_metrics.

    Returns list of per-graph result dicts with depth distribution info.
    """
    truth = batch[output_key]
    preds = output[output_key]
    edge_index = batch.edge_index
    results = []

    for g in range(batch.num_graphs):
        node_mask = (batch.batch == g)
        global_ids = torch.where(node_mask)[0]
        n_nodes = global_ids.size(0)
        offset = global_ids[0].item()

        # Per-node correctness using the SAME logic as calc_metrics
        node_correct_list = []
        gt_pred_array = np.arange(n_nodes)  # for depth computation

        for local_j in range(n_nodes):
            global_j = offset + local_j
            idx = (edge_index[0] == global_j)
            if idx.sum() == 0:
                node_correct_list.append(True)
                continue

            pred_argmax = preds[idx].argmax(dim=-1).item()
            truth_argmax = truth[idx].argmax(dim=-1).item()
            node_correct_list.append(pred_argmax == truth_argmax)

            # Extract ground-truth predecessor for depth computation
            targets = edge_index[1, idx]
            gt_pred_array[local_j] = targets[truth_argmax].item() - offset

        node_correct = np.array(node_correct_list)
        graph_correct = node_correct.all()

        # Per-node depths from ground-truth BFS tree
        gt_depths = node_depths_from_predecessors(gt_pred_array)
        max_depth = int(gt_depths.max()) if n_nodes > 0 else 0

        # Collect depths at which mistakes occur
        mistake_depths = gt_depths[~node_correct].tolist() if not graph_correct else []

        results.append({
            "n_nodes": n_nodes,
            "max_depth": max_depth,
            "graph_correct": bool(graph_correct),
            "n_wrong_nodes": int((~node_correct).sum()),
            "mistake_depths": mistake_depths,
            "all_depths": gt_depths.tolist(),
            "node_accuracy": float(node_correct.mean()),
        })

    return results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(name, results):
    correct = [r for r in results if r["graph_correct"]]
    wrong   = [r for r in results if not r["graph_correct"]]

    print(f"\n{'=' * 64}")
    print(f"  {name}")
    print(f"{'=' * 64}")
    print(f"  Total graphs         : {len(results)}")
    print(f"  Correct predictions  : {len(correct)}  ({100*len(correct)/len(results):.1f}%)")
    print(f"  Wrong predictions    : {len(wrong)}  ({100*len(wrong)/len(results):.1f}%)")

    # Node accuracy
    all_node_acc = [r["node_accuracy"] for r in results]
    print(f"  Avg node accuracy    : {np.mean(all_node_acc):.4f}")
    print()

    # --- avg depth ---
    if correct:
        d = [r["max_depth"] for r in correct]
        print(f"  Avg BFS depth (correct graphs)  : {np.mean(d):.2f}  (std {np.std(d):.2f})")
    else:
        print(f"  Avg BFS depth (correct graphs)  : N/A")
    if wrong:
        d = [r["max_depth"] for r in wrong]
        print(f"  Avg BFS depth (wrong graphs)    : {np.mean(d):.2f}  (std {np.std(d):.2f})")
    else:
        print(f"  Avg BFS depth (wrong graphs)    : N/A")

    # --- mistake depth distribution ---
    if wrong:
        all_mistake_depths = []
        all_graph_depths = []
        for r in wrong:
            all_mistake_depths.extend(r["mistake_depths"])
            all_graph_depths.extend(r["all_depths"])

        if all_mistake_depths:
            depth_counts = defaultdict(int)
            for d in all_mistake_depths:
                depth_counts[d] += 1

            total_depth_counts = defaultdict(int)
            for d in all_graph_depths:
                total_depth_counts[d] += 1

            max_d = max(depth_counts.keys())

            # Avg wrong nodes per wrong graph
            avg_wrong_nodes = np.mean([r["n_wrong_nodes"] for r in wrong])
            print(f"\n  Avg wrong nodes / wrong graph : {avg_wrong_nodes:.1f}")

            print(f"\n  Mistake depth distribution (across all wrong graphs):")
            print(f"  {'Depth':>6}  {'Total N':>7}  {'Mistks':>7}  {'% Error':>7}  Bar")
            print(f"  {'-'*6}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*30}")
            total_mistakes = len(all_mistake_depths)
            for d in range(max_d + 1):
                c = depth_counts.get(d, 0)
                tot_n = total_depth_counts.get(d, 0)
                pct_err = 100.0 * c / tot_n if tot_n else 0.0
                pct_all_mistakes = 100.0 * c / total_mistakes if total_mistakes else 0
                bar = "█" * int(pct_all_mistakes / 2)
                print(f"  {d:>6}  {tot_n:>7}  {c:>7}  {pct_err:>6.1f}%  {bar}")

    print(f"{'=' * 64}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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

    # --- Find output key and type ---
    specs = model.specs
    output_key = None
    output_type = None
    for k, v in specs.items():
        stage = v[0]
        stage_name = stage.name if hasattr(stage, "name") else str(stage)
        if stage_name.upper() == "OUTPUT":
            output_key = k
            output_type = v[2]  # e.g. "pointer"
            break
    assert output_key is not None, "Could not find OUTPUT key in specs"
    print(f"Output key: {output_key} (type: {output_type})\n")

    ignore_hints = (cfg.TRAIN.LOSS.HINT_LOSS_WEIGHT == 0.0)
    # Use the same root path as get_dataset() to avoid stale cached data
    data_root = os.path.join(cfg.DATA.ROOT, "salsaclrs", f"seed_{args.seed}")
    os.makedirs(data_root, exist_ok=True)

    # --- Determine autocast context to match pl.Trainer(precision=...) ---
    precision = cfg.TRAIN.PRECISION
    if precision in ("16-mixed", "16"):
        autocast_ctx = torch.autocast(device.type, dtype=torch.float16)
        print(f"Using mixed precision: float16 (matching cfg.TRAIN.PRECISION={precision})")
    elif precision in ("bf16-mixed", "bf16"):
        autocast_ctx = torch.autocast(device.type, dtype=torch.bfloat16)
        print(f"Using mixed precision: bfloat16 (matching cfg.TRAIN.PRECISION={precision})")
    else:
        autocast_ctx = contextlib.nullcontext()
        print(f"Using full precision fp32 (cfg.TRAIN.PRECISION={precision})")

    # --- Evaluate each path depth config ---
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
        with torch.no_grad(), autocast_ctx:
            for loader in datamodule.test_dataloader():
                for batch in loader:
                    batch = batch.to(device)
                    output, hints, hidden = model(batch)
                    all_results.extend(analyse_batch(batch, output, output_key, output_type))

        print_report(config_name, all_results)


if __name__ == '__main__':
    main()
