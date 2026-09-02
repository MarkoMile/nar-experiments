"""
BFS Depth Analysis on various graph families.

Rolls out a trained model on multiple graph types (WS-1600, ER-1600, Delaunay-1600)
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
import json
import argparse
import contextlib

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

# ER edge probability is scaled ~1/N so mean degree stays constant across
# sizes, matching the SALSA-CLRS eval protocol in bestmodel.yml. (This is why
# these graphs are not "distance <= 2" small-world: diameter grows as log N.)
ER_P_RANGE = {
    800: [0.008, 0.025],
    1600: [0.004, 0.0125],
}


def build_graph_configs(sizes, families, ws_p_sweep=None, ws_k=6):
    """Build the {name: generator kwargs} registry to evaluate."""
    configs = {}
    for n in sizes:
        for fam in families:
            if fam == "ws":
                configs[f"ws_{n}"] = {
                    "graph_generator": "ws",
                    "graph_generator_kwargs": {"p_range": [0.05, 0.2], "k": [4, 6, 8], "n": n},
                }
            elif fam == "er":
                if n not in ER_P_RANGE:
                    raise ValueError(f"No ER p_range calibrated for n={n}; add one to ER_P_RANGE.")
                configs[f"er_{n}"] = {
                    "graph_generator": "er",
                    "graph_generator_kwargs": {"p_range": ER_P_RANGE[n], "n": n},
                }
            elif fam == "delaunay":
                configs[f"delaunay_{n}"] = {
                    "graph_generator": "delaunay",
                    "graph_generator_kwargs": {"n": n},
                }
            else:
                raise ValueError(f"Unknown graph family {fam!r}")

    # WS rewiring sweep: identical ring-lattice base and degree, one variable
    # (the rewiring probability p). p=0 is a pure ring lattice with no shortcuts.
    for n in sizes:
        for p in (ws_p_sweep or []):
            configs[f"wsp_{n}_p{int(round(p * 1000)):04d}"] = {
                "graph_generator": "ws",
                "graph_generator_kwargs": {"p": float(p), "k": int(ws_k), "n": n},
            }
    return configs


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


def shortcut_stats(local_edges, gt_pred, node_correct, n_nodes, k):
    """Localise errors relative to Watts-Strogatz shortcut edges.

    WS graphs are a rewired ring lattice, and networkx keeps the ring ordering in
    the node labels, so an edge is a rewired "shortcut" exactly when its endpoints
    are further apart around the ring than k // 2. This lets us ask whether wrong
    nodes are the ones whose true BFS parent edge is a shortcut, rather than only
    whether shortcut-heavy graphs fail.
    """
    half = max(int(k) // 2, 1)

    def is_shortcut(u, v):
        d = abs(int(u) - int(v))
        return min(d, n_nodes - d) > half

    incident = np.zeros(n_nodes, dtype=bool)
    n_shortcut_edges = 0
    for u, v in local_edges:
        if is_shortcut(u, v):
            n_shortcut_edges += 1
            incident[u] = True
            incident[v] = True

    parent_shortcut = np.zeros(n_nodes, dtype=bool)
    for v in range(n_nodes):
        pv = gt_pred[v]
        if pv != v and is_shortcut(v, pv):
            parent_shortcut[v] = True

    def counts(mask):
        return int(mask.sum()), int((mask & ~node_correct).sum())

    ps_tot, ps_wrong = counts(parent_shortcut)
    pr_tot, pr_wrong = counts(~parent_shortcut)
    in_tot, in_wrong = counts(incident)
    ni_tot, ni_wrong = counts(~incident)
    return {
        "k_est": int(k),
        # local_edges holds both directions, so halve to get undirected count.
        "n_shortcut_edges": n_shortcut_edges // 2,
        "parent_shortcut_total": ps_tot, "parent_shortcut_wrong": ps_wrong,
        "parent_ring_total": pr_tot, "parent_ring_wrong": pr_wrong,
        "incident_shortcut_total": in_tot, "incident_shortcut_wrong": in_wrong,
        "no_shortcut_total": ni_tot, "no_shortcut_wrong": ni_wrong,
    }


def analyse_batch(batch, output, output_key, device, detect_shortcuts=False):
    """Analyse one batch. Returns list of per-graph result dicts."""
    truth = batch[output_key]
    preds = output[output_key]
    edge_index = batch.edge_index
    results = []

    batch_degrees = torch.bincount(edge_index[0], minlength=batch.num_nodes)

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

        graph_degrees = batch_degrees[global_ids].cpu().numpy()

        record = {
            "n_nodes": n_nodes,
            "max_depth": max_depth,
            "graph_correct": bool(graph_correct),
            "n_wrong_nodes": int((~node_correct).sum()),
            "mistake_depths": mistake_depths,
            "all_depths": gt_depths.tolist(),
            "avg_graph_degree": float(graph_degrees.mean()) if n_nodes > 0 else 0.0,
            "correct_degrees": graph_degrees[node_correct].tolist(),
            "incorrect_degrees": graph_degrees[~node_correct].tolist(),
        }

        if detect_shortcuts and n_nodes > 0:
            edge_mask = node_mask[edge_index[0]]
            local_edges = (edge_index[:, edge_mask] - offset).cpu().numpy().T
            # WS rewiring preserves edge count, so mean degree recovers k -- but
            # SALSA-CLRS's adjacency carries a self-loop per node, so subtract the
            # self-loops rather than relying on integer division to absorb them.
            n_self_loops = int((local_edges[:, 0] == local_edges[:, 1]).sum())
            mean_deg = float(graph_degrees.mean()) - (n_self_loops / n_nodes)
            k_est = max(int(round(mean_deg)), 2)
            record.update(shortcut_stats(local_edges, gt_pred, node_correct, n_nodes, k_est))

        results.append(record)

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
    
    all_degrees = [d for r in results for d in r.get("correct_degrees", []) + r.get("incorrect_degrees", [])]
    if all_degrees:
        print(f"  Total avg node degree: {np.mean(all_degrees):.2f}  (std {np.std(all_degrees):.2f})")
    print()

    # --- avg depth & degree ---
    if correct:
        d = [r["max_depth"] for r in correct]
        deg = [r["avg_graph_degree"] for r in correct]
        print(f"  Avg BFS depth (correct graphs)  : {np.mean(d):.2f}  (std {np.std(d):.2f})")
        print(f"  Avg node degree (correct graphs): {np.mean(deg):.2f}")
    else:
        print(f"  Avg BFS depth (correct graphs)  : N/A")
        print(f"  Avg node degree (correct graphs): N/A")
    if wrong:
        d = [r["max_depth"] for r in wrong]
        deg = [r["avg_graph_degree"] for r in wrong]
        print(f"  Avg BFS depth (wrong graphs)    : {np.mean(d):.2f}  (std {np.std(d):.2f})")
        print(f"  Avg node degree (wrong graphs)  : {np.mean(deg):.2f}")
    else:
        print(f"  Avg BFS depth (wrong graphs)    : N/A")
        print(f"  Avg node degree (wrong graphs)  : N/A")

    print()
    all_corr_nod_deg = [d for r in results for d in r.get("correct_degrees", [])]
    all_incorr_nod_deg = [d for r in results for d in r.get("incorrect_degrees", [])]

    if all_corr_nod_deg:
        print(f"  Avg node degree (correct nodes)   : {np.mean(all_corr_nod_deg):.2f}  (std {np.std(all_corr_nod_deg):.2f})")
    else:
        print(f"  Avg node degree (correct nodes)   : N/A")

    if all_incorr_nod_deg:
        print(f"  Avg node degree (incorrect nodes) : {np.mean(all_incorr_nod_deg):.2f}  (std {np.std(all_incorr_nod_deg):.2f})")
    else:
        print(f"  Avg node degree (incorrect nodes) : N/A")

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

    # --- shortcut localisation (WS only) ---
    if results and "parent_shortcut_total" in results[0]:
        agg = {k: sum(r.get(k, 0) for r in results) for k in (
            "n_shortcut_edges",
            "parent_shortcut_total", "parent_shortcut_wrong",
            "parent_ring_total", "parent_ring_wrong",
            "incident_shortcut_total", "incident_shortcut_wrong",
            "no_shortcut_total", "no_shortcut_wrong",
        )}

        def pct(wrong, total):
            return f"{100.0 * wrong / total:.2f}%" if total else "N/A"

        print(f"\n  Shortcut localisation (k~{results[0].get('k_est', '?')}, "
              f"{agg['n_shortcut_edges']} shortcut edges across {len(results)} graphs):")
        print(f"  {'node group':<34}{'nodes':>9}{'wrong':>8}{'err rate':>10}")
        print(f"  {'-' * 34}{'-' * 9}{'-' * 8}{'-' * 10}")
        for label, tk, wk in (
            ("true BFS parent IS a shortcut", "parent_shortcut_total", "parent_shortcut_wrong"),
            ("true BFS parent is a ring edge", "parent_ring_total", "parent_ring_wrong"),
            ("incident to >=1 shortcut", "incident_shortcut_total", "incident_shortcut_wrong"),
            ("incident to no shortcut", "no_shortcut_total", "no_shortcut_wrong"),
        ):
            print(f"  {label:<34}{agg[tk]:>9}{agg[wk]:>8}{pct(agg[wk], agg[tk]):>10}")

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
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--precision", type=str, default=None,
        choices=["32", "16-mixed", "bf16-mixed"],
        help="Override eval precision. Default: fp32 (this script runs the model "
             "directly and does NOT inherit cfg.TRAIN.PRECISION)."
    )
    parser.add_argument("--sizes", type=int, nargs="+", default=[1600],
                        help="Graph sizes to evaluate. Default: 1600")
    parser.add_argument("--families", type=str, nargs="*", default=["ws", "er", "delaunay"],
                        help="Graph families. Default: ws er delaunay. Pass --families with no "
                             "values to evaluate only the --ws-p-sweep bins.")
    parser.add_argument("--ws-p-sweep", type=float, nargs="*", default=None,
                        help="Rewiring probabilities for the WS sweep, e.g. 0 0.01 0.05 0.1 0.2 0.5 1.0. "
                             "Uses a fixed k so p is the only variable.")
    parser.add_argument("--ws-k", type=int, default=6, help="Fixed k for the WS sweep. Default: 6")
    parser.add_argument("--max-cores", type=int, default=-1,
                        help="Cores for graph generation. -1 (default) is serial.")
    parser.add_argument("--mask-mode", type=str, default=None, choices=["soft", "hard"],
                        help="Override MODEL.AUTOREGRESSIVE.MASK_MODE. 'hard' thresholds the "
                             "re-injected mask hint at 0.5 during rollout (inference only).")
    parser.add_argument("--shortcuts", action="store_true",
                        help="Localise errors relative to WS shortcut edges (WS configs only).")
    parser.add_argument("--json-out", type=str, default=None,
                        help="Write per-graph results to this JSON file.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # --- Config & device ---
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # This script calls the model directly rather than through a Lightning
    # Trainer, so without an explicit override it runs in full fp32 -- unlike
    # eval_checkpoint.py, which applies cfg.TRAIN.PRECISION (16-mixed).
    precision = args.precision or "32"
    if precision in ("16-mixed", "16"):
        autocast_ctx = torch.autocast(device.type, dtype=torch.float16)
        print(f"Precision: float16 mixed (from --precision, value={precision})")
    elif precision in ("bf16-mixed", "bf16"):
        autocast_ctx = torch.autocast(device.type, dtype=torch.bfloat16)
        print(f"Precision: bfloat16 mixed (from --precision, value={precision})")
    else:
        autocast_ctx = contextlib.nullcontext()
        print(f"Precision: fp32 (default for this script)")

    # --- Load model ---
    print(f"Loading checkpoint: {args.ckpt}")
    model = SALSACLRSModel.load_from_checkpoint(args.ckpt, map_location=device, strict=False)
    cfg = model.cfg
    if args.mask_mode is not None:
        # Old checkpoints predate this key, so allow it to be added.
        cfg.MODEL.AUTOREGRESSIVE.set_new_allowed(True)
        cfg.MODEL.AUTOREGRESSIVE.MASK_MODE = args.mask_mode
        print(f"AR mask hint mode: {args.mask_mode}")
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
    # Seeded root so different seeds do not reuse each other's cached graphs.
    data_root = os.path.join(cfg.DATA.ROOT, "salsaclrs", f"seed_{args.seed}")
    os.makedirs(data_root, exist_ok=True)

    graph_configs = build_graph_configs(
        sizes=args.sizes,
        families=args.families,
        ws_p_sweep=args.ws_p_sweep,
        ws_k=args.ws_k,
    )
    print(f"Evaluating {len(graph_configs)} config(s): {', '.join(graph_configs)}\n")

    collected = {}

    # --- Evaluate each graph family ---
    for config_name, gcfg in graph_configs.items():
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
            max_cores=args.max_cores,
        )

        datamodule = SALSACLRSDataModule(
            test_datasets=[dataset],
            batch_size=args.batch_size,
            num_workers=0,
            test_batch_size=args.batch_size,
        )

        detect_shortcuts = args.shortcuts and gcfg["graph_generator"] == "ws"

        all_results = []
        with torch.no_grad(), autocast_ctx:
            for loader in datamodule.test_dataloader():
                for batch in loader:
                    batch = batch.to(device)
                    output, hints, hidden = model(batch)
                    all_results.extend(
                        analyse_batch(batch, output, output_key, device,
                                      detect_shortcuts=detect_shortcuts)
                    )

        print_report(config_name, all_results)
        collected[config_name] = all_results

    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        payload = {
            "ckpt": args.ckpt,
            "seed": args.seed,
            "precision": precision,
            "num_samples": args.num_samples,
            "configs": {
                name: {
                    "n_graphs": len(rs),
                    "n_correct": sum(1 for r in rs if r["graph_correct"]),
                    "graph_accuracy": (sum(1 for r in rs if r["graph_correct"]) / len(rs)) if rs else None,
                    "graphs": rs,
                }
                for name, rs in collected.items()
            },
        }
        with open(args.json_out, "w") as f:
            json.dump(payload, f)
        print(f"\nWrote results to {args.json_out}")


if __name__ == "__main__":
    main()
