"""
Turn the rebuttal suite's JSON output into the tables the rebuttal quotes.

Reads results/rebuttal/*.json (written by tests/bfs_depth_analysis.py --json-out
via tests/run_rebuttal_suite.py) and prints, per stage:

  depth    graph accuracy + where mistakes land, per checkpoint and precision.
           The rebuttal's central claim is that Delaunay/ER wrong graphs contain
           ~1 wrong node at the *terminal* BFS level (algorithmic execution with
           an occasional frontier slip) while WS wrong graphs contain many, mid
           rollout (heuristic-like) -- so both numbers are reported.
  wssweep  graph accuracy vs rewiring p, plus error rate for nodes whose true
           BFS parent is a shortcut vs a ring edge.
  disc     soft vs hard re-injected mask hint.

Usage:
    python tests/summarize_rebuttal.py --results-dir results/rebuttal
"""

import os
import re
import glob
import json
import argparse
from collections import defaultdict

import numpy as np


def load(results_dir):
    out = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        if os.path.basename(path) == "run_summary.json":
            continue
        with open(path) as f:
            payload = json.load(f)
        payload["_file"] = os.path.basename(path)
        out.append(payload)
    return out


def terminal_share(graphs):
    """Fraction of mistakes landing at each graph's deepest BFS level.

    A model that executes BFS and slips at the frontier puts its errors at the
    last level; a model running a heuristic scatters them through the rollout.
    """
    at_end = total = 0
    for g in graphs:
        if g["graph_correct"]:
            continue
        for d in g["mistake_depths"]:
            total += 1
            if d >= g["max_depth"]:
                at_end += 1
    return (at_end / total) if total else None


def summarize_configs(payload):
    rows = []
    for name, c in payload["configs"].items():
        graphs = c["graphs"]
        wrong = [g for g in graphs if not g["graph_correct"]]
        n = len(graphs)
        acc = c["graph_accuracy"]
        node_acc = None
        tot_nodes = sum(g["n_nodes"] for g in graphs)
        if tot_nodes:
            node_acc = 1.0 - sum(g["n_wrong_nodes"] for g in graphs) / tot_nodes
        rows.append({
            "config": name,
            "n": n,
            "correct": c["n_correct"],
            "graph_acc": acc,
            "node_acc": node_acc,
            "wrong_nodes_per_wrong_graph": (np.mean([g["n_wrong_nodes"] for g in wrong])
                                            if wrong else None),
            "depth_correct": (np.mean([g["max_depth"] for g in graphs if g["graph_correct"]])
                              if c["n_correct"] else None),
            "depth_wrong": (np.mean([g["max_depth"] for g in wrong]) if wrong else None),
            "terminal_share": terminal_share(graphs),
        })
    return rows


def fmt(v, spec=".4f"):
    return "n/a" if v is None else format(v, spec)


def print_depth(payloads):
    print("\n" + "=" * 108)
    print("DEPTH / MISTAKE LOCALISATION")
    print("=" * 108)
    hdr = (f"{'ckpt':<22}{'prec':>12}  {'config':<18}{'n':>5}{'graph acc':>11}"
           f"{'node acc':>11}{'wrong/wrong g':>15}{'depth ok':>10}{'depth bad':>11}{'terminal':>10}")
    print(hdr)
    print("-" * 108)
    for p in payloads:
        ck = os.path.basename(p["ckpt"]).replace(".ckpt", "")
        for r in summarize_configs(p):
            print(f"{ck:<22}{p['precision']:>12}  {r['config']:<18}{r['n']:>5}"
                  f"{fmt(r['graph_acc']):>11}{fmt(r['node_acc'], '.5f'):>11}"
                  f"{fmt(r['wrong_nodes_per_wrong_graph'], '.1f'):>15}"
                  f"{fmt(r['depth_correct'], '.1f'):>10}{fmt(r['depth_wrong'], '.1f'):>11}"
                  f"{fmt(r['terminal_share'], '.0%'):>10}")


def print_precision_ab(payloads):
    """Same checkpoint, same config, fp32 vs reduced precision."""
    table = defaultdict(dict)
    for p in payloads:
        ck = os.path.basename(p["ckpt"]).replace(".ckpt", "")
        for r in summarize_configs(p):
            table[(ck, r["config"])][p["precision"]] = r["graph_acc"]
    precs = sorted({pr for v in table.values() for pr in v})
    if len(precs) < 2:
        return
    print("\n" + "=" * 108)
    print("PRECISION A/B  (graph accuracy)")
    print("=" * 108)
    print(f"{'ckpt':<22}{'config':<18}" + "".join(f"{p:>14}" for p in precs) + f"{'delta':>10}")
    print("-" * 108)
    for (ck, cfg), v in sorted(table.items()):
        vals = [v.get(p) for p in precs]
        delta = (vals[0] - vals[-1]) if all(x is not None for x in (vals[0], vals[-1])) else None
        print(f"{ck:<22}{cfg:<18}" + "".join(f"{fmt(x):>14}" for x in vals)
              + f"{fmt(delta, '+.4f'):>10}")


def print_wssweep(payloads):
    sweep = [p for p in payloads if any(k.startswith("wsp_") for k in p["configs"])]
    if not sweep:
        return
    print("\n" + "=" * 108)
    print("WS REWIRING SWEEP  (fixed k, p is the only variable; p=0 is a pure ring lattice)")
    print("=" * 108)
    print(f"{'ckpt':<22}{'p':>7}{'n':>5}{'graph acc':>11}{'node acc':>11}"
          f"{'depth':>8}{'err|parent=shortcut':>21}{'err|parent=ring':>18}")
    print("-" * 108)
    for p in sweep:
        ck = os.path.basename(p["ckpt"]).replace(".ckpt", "")
        for name, c in sorted(p["configs"].items()):
            m = re.match(r"wsp_(\d+)_p(\d+)", name)
            if not m:
                continue
            graphs = c["graphs"]
            tot_nodes = sum(g["n_nodes"] for g in graphs)
            node_acc = 1.0 - sum(g["n_wrong_nodes"] for g in graphs) / tot_nodes if tot_nodes else None
            depth = np.mean([g["max_depth"] for g in graphs]) if graphs else None
            agg = {k: sum(g.get(k, 0) for g in graphs) for k in
                   ("parent_shortcut_total", "parent_shortcut_wrong",
                    "parent_ring_total", "parent_ring_wrong")}
            sc = (agg["parent_shortcut_wrong"] / agg["parent_shortcut_total"]
                  if agg["parent_shortcut_total"] else None)
            rg = (agg["parent_ring_wrong"] / agg["parent_ring_total"]
                  if agg["parent_ring_total"] else None)
            print(f"{ck:<22}{int(m.group(2)) / 1000:>7.3f}{len(graphs):>5}"
                  f"{fmt(c['graph_accuracy']):>11}{fmt(node_acc, '.5f'):>11}"
                  f"{fmt(depth, '.1f'):>8}{fmt(sc, '.3%'):>21}{fmt(rg, '.3%'):>18}")


def print_disc(payloads):
    modes = {}
    for p in payloads:
        m = re.search(r"disc_.*_(soft|hard)\.json$", p["_file"])
        if not m:
            continue
        ck = os.path.basename(p["ckpt"]).replace(".ckpt", "")
        for r in summarize_configs(p):
            modes.setdefault((ck, r["config"]), {})[m.group(1)] = r
    if not modes:
        return
    print("\n" + "=" * 108)
    print("DISCRETISED MASK HINT  (inference only: threshold re-injected reach_h at 0.5)")
    print("=" * 108)
    print(f"{'ckpt':<22}{'config':<18}{'soft acc':>11}{'hard acc':>11}{'delta':>10}"
          f"{'soft wrong/g':>14}{'hard wrong/g':>14}")
    print("-" * 108)
    for (ck, cfg), v in sorted(modes.items()):
        s, h = v.get("soft"), v.get("hard")
        if not (s and h):
            continue
        delta = h["graph_acc"] - s["graph_acc"]
        print(f"{ck:<22}{cfg:<18}{fmt(s['graph_acc']):>11}{fmt(h['graph_acc']):>11}"
              f"{fmt(delta, '+.4f'):>10}"
              f"{fmt(s['wrong_nodes_per_wrong_graph'], '.1f'):>14}"
              f"{fmt(h['wrong_nodes_per_wrong_graph'], '.1f'):>14}")


def main():
    ap = argparse.ArgumentParser(description="Summarize the rebuttal suite results")
    ap.add_argument("--results-dir", type=str, default="results/rebuttal")
    args = ap.parse_args()

    payloads = load(args.results_dir)
    if not payloads:
        raise SystemExit(f"No result JSON in {args.results_dir}")
    print(f"Loaded {len(payloads)} result file(s) from {args.results_dir}")

    depth = [p for p in payloads if p["_file"].startswith("depth_")]
    print_depth(depth or payloads)
    print_precision_ab(depth or payloads)
    print_wssweep(payloads)
    print_disc(payloads)
    print()


if __name__ == "__main__":
    main()
