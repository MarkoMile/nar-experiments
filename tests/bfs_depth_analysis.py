"""
BFS Depth Analysis on various graph families.

Rolls out a trained model on multiple graph types (WS-80, WS-800, ER-800)
and reports:
  - Average BFS tree depth of correct vs incorrect graphs.
  - At which BFS depths the prediction mistakes occur.

Usage:
    python tests/bfs_depth_analysis.py \
        --cfg src/configs/bfs/PGN-grok-hl.yml \
        --ckpt data/checkpoints/bfs/PGN/seed42-final.ckpt \
        --num-samples 32
"""

import os
import sys
import argparse

import torch
import numpy as np
from collections import defaultdict
from loguru import logger

# Configure loguru to suppress DEBUG messages
logger.remove()
logger.add(sys.stderr, level="INFO")

# Add project root so absolute imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.module import SALSACLRSModel
from src.utils.graph_generation import get_dataset  # triggers torch.load monkeypatch

from salsaclrs import SALSACLRSDataset, SALSACLRSDataModule


# ---------------------------------------------------------------------------
# Graph configs to evaluate
# ---------------------------------------------------------------------------

GRAPH_CONFIGS = {
    "ws_80": {
        "graph_generator": "ws",
        "graph_generator_kwargs": {"p_range": [0.05, 0.2], "k": [4, 6, 8], "n": 80},
    },
    "ws_800": {
        "graph_generator": "ws",
        "graph_generator_kwargs": {"p_range": [0.05, 0.2], "k": [4, 6, 8], "n": 800},
    },
    "er_800": {
        "graph_generator": "er",
        "graph_generator_kwargs": {"p_range": [0.008, 0.025], "n": 800},
    },
    "delaunay_800": {
        "graph_generator": "delaunay",
        "graph_generator_kwargs": {"n": 800},
    },
    "ws_1600": {
        "graph_generator": "ws",
        "graph_generator_kwargs": {"p_range": [0.05, 0.2], "k": [4, 6, 8], "n": 1600},
    },
    "er_1600": {
        "graph_generator": "er",
        "graph_generator_kwargs": {"p_range": [0.0046, 0.014], "n": 1600},
    },
    "delaunay_1600": {
        "graph_generator": "delaunay",
        "graph_generator_kwargs": {"n": 1600},
    },
}


# ---------------------------------------------------------------------------
# Helpers
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


def extract_graph_pointers(edge_index, truth_or_preds, n_nodes, offset):
    """Extract a dense predecessor array for one graph from sparse one-hot edge data."""
    pred = np.arange(n_nodes)
    for local_j in range(n_nodes):
        global_j = offset + local_j
        idx = (edge_index[0] == global_j)
        if idx.sum() == 0:
            continue
        targets = edge_index[1, idx]
        vals = truth_or_preds[idx]
        chosen = vals.argmax(dim=-1)
        pred[local_j] = targets[chosen].item() - offset
    return pred


def analyse_batch(batch, output, output_key, device):
    """Analyse one batch. Returns list of per-graph result dicts."""
    truth = batch[output_key]
    preds = output[output_key]
    edge_index = batch.edge_index
    results = []

    for g in range(batch.num_graphs):
        node_mask = (batch.batch == g)
        global_ids = torch.where(node_mask)[0]
        n_nodes = global_ids.size(0)
        offset = global_ids[0].item()

        # Dense predecessor arrays
        gt_pred = extract_graph_pointers(edge_index, truth, n_nodes, offset)
        md_pred = extract_graph_pointers(edge_index, preds, n_nodes, offset)

        # Per-node depths from ground-truth BFS tree
        gt_depths = node_depths_from_predecessors(gt_pred)
        max_depth = int(gt_depths.max()) if n_nodes > 0 else 0

        # Per-node correctness
        node_correct = (gt_pred == md_pred)
        graph_correct = node_correct.all()

        # Collect depths at which mistakes occur
        mistake_depths = gt_depths[~node_correct].tolist() if not graph_correct else []

        results.append({
            "n_nodes": n_nodes,
            "max_depth": max_depth,
            "graph_correct": bool(graph_correct),
            "n_wrong_nodes": int((~node_correct).sum()),
            "mistake_depths": mistake_depths,
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
    print(f"  Correct predictions  : {len(correct)}")
    print(f"  Wrong predictions    : {len(wrong)}")
    print()

    # --- avg depth ---
    if correct:
        d = [r["max_depth"] for r in correct]
        print(f"  Avg BFS depth (correct) : {np.mean(d):.2f}  (std {np.std(d):.2f})")
    else:
        print(f"  Avg BFS depth (correct) : N/A")
    if wrong:
        d = [r["max_depth"] for r in wrong]
        print(f"  Avg BFS depth (wrong)   : {np.mean(d):.2f}  (std {np.std(d):.2f})")
    else:
        print(f"  Avg BFS depth (wrong)   : N/A")

    # --- mistake depth distribution ---
    if wrong:
        all_mistake_depths = []
        for r in wrong:
            all_mistake_depths.extend(r["mistake_depths"])

        if all_mistake_depths:
            depth_counts = defaultdict(int)
            for d in all_mistake_depths:
                depth_counts[d] += 1
            max_d = max(depth_counts.keys())

            # Avg wrong nodes per wrong graph
            avg_wrong_nodes = np.mean([r["n_wrong_nodes"] for r in wrong])
            print(f"\n  Avg wrong nodes / wrong graph : {avg_wrong_nodes:.1f}")

            print(f"\n  Mistake depth distribution (across all wrong graphs):")
            print(f"  {'Depth':>6}  {'Count':>7}  {'%':>7}  Bar")
            print(f"  {'-'*6}  {'-'*7}  {'-'*7}  {'-'*30}")
            total = len(all_mistake_depths)
            for d in range(max_d + 1):
                c = depth_counts.get(d, 0)
                pct = 100.0 * c / total if total else 0
                bar = "█" * int(pct / 2)
                print(f"  {d:>6}  {c:>7}  {pct:>6.1f}%  {bar}")

    print(f"{'=' * 64}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="BFS depth analysis")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--num-samples", type=int, default=32, help="Graphs per config")
    parser.add_argument("--batch-size", type=int, default=1, help="Inference batch size")
    parser.add_argument("--device", type=str, default="auto", help="cpu / cuda / auto")
    args = parser.parse_args()

    # --- Config & device ---
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

    # --- Evaluate each graph family ---
    for config_name, gcfg in GRAPH_CONFIGS.items():
        print(f"▸ {config_name}: generating {args.num_samples} samples …")

        dataset = SALSACLRSDataset(
            root=data_root,
            split="test",
            algorithm=cfg.ALGORITHM,
            num_samples=args.num_samples,
            graph_generator=gcfg["graph_generator"],
            graph_generator_kwargs=gcfg["graph_generator_kwargs"],
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


if __name__ == "__main__":
    main()
