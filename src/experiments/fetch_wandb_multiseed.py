"""
Download the multiseed runs from Weights & Biases.

Pulls run summaries and full metric histories for every run in a W&B group, so
the per-seed table and the grokking figures can be rebuilt offline without
re-running anything on a GPU.

Usage:
    python src/experiments/fetch_wandb_multiseed.py \
        --entity markomile-petnica \
        --project nar-experiments-finetuning \
        --group multiseed \
        --out-dir results/wandb/multiseed
"""

import os
import re
import sys
import json
import argparse

import pandas as pd
from loguru import logger


# Metrics whose full history we want (for the weight-norm / val-accuracy curves
# that the grokking split is read off).
HISTORY_KEYS = [
    "trainer/global_step",
    "train/weight_norm",
    "val/node_accuracy/ws_800",
    "val/graph_accuracy/ws_800",
    "val/node_accuracy/delaunay_800",
    "val/graph_accuracy/delaunay_800",
]

SEED_RE = re.compile(r"(?:seed[-_]?|eval-)(\d+)")


def infer_seed(run):
    """Best-effort seed extraction from config, then from the run name."""
    for key in ("seed", "SEED", "TRAIN.SEED"):
        if key in run.config:
            try:
                return int(run.config[key])
            except (TypeError, ValueError):
                pass
    m = SEED_RE.search(run.name or "")
    return int(m.group(1)) if m else None


def main():
    parser = argparse.ArgumentParser(description="Fetch W&B multiseed runs")
    parser.add_argument("--entity", type=str, default="markomile-petnica")
    parser.add_argument("--project", type=str, default="nar-experiments-finetuning")
    parser.add_argument("--group", type=str, default="multiseed")
    parser.add_argument("--out-dir", type=str, default="results/wandb/multiseed")
    parser.add_argument("--history-samples", type=int, default=100000,
                        help="Max history rows per run. Large enough to be effectively full.")
    args = parser.parse_args()

    try:
        import wandb
    except ImportError:
        sys.exit("wandb is not installed. pip install wandb")

    os.makedirs(args.out_dir, exist_ok=True)
    hist_dir = os.path.join(args.out_dir, "history")
    os.makedirs(hist_dir, exist_ok=True)

    api = wandb.Api()
    path = f"{args.entity}/{args.project}"
    logger.info(f"Querying {path} for group={args.group!r}")
    runs = list(api.runs(path, filters={"group": args.group}))
    if not runs:
        sys.exit(f"No runs found in {path} with group={args.group!r}. "
                 f"Check the entity/project/group names.")
    logger.info(f"Found {len(runs)} run(s)")

    summary_rows = []
    for run in runs:
        seed = infer_seed(run)
        logger.info(f"  {run.name} (seed={seed}, state={run.state})")

        row = {"run_id": run.id, "run_name": run.name, "seed": seed, "state": run.state}
        # summary holds the last logged value of every metric
        for k, v in run.summary.items():
            if isinstance(v, (int, float)) and not k.startswith("_"):
                row[k] = v
        summary_rows.append(row)

        # Ask for the curve keys directly; a metric logged only during training
        # (e.g. train/weight_norm) does not appear in run.summary, so filtering
        # on summary keys silently returns nothing.
        hist = None
        try:
            hist = run.history(keys=HISTORY_KEYS, samples=args.history_samples, pandas=True)
        except Exception as e:  # noqa: BLE001 - one bad run must not lose the rest
            logger.warning(f"    keyed history failed ({e}); falling back to full history")
        if hist is None or len(hist) == 0:
            try:
                hist = run.history(samples=args.history_samples, pandas=True)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"    history fetch failed: {e}")
                continue
        if hist is None or len(hist) == 0:
            logger.warning("    no history rows")
            continue
        out = os.path.join(hist_dir, f"{run.name or run.id}.csv")
        hist.to_csv(out, index=False)
        logger.info(f"    wrote {len(hist)} history rows -> {out}")

    summary = pd.DataFrame(summary_rows).sort_values("seed", na_position="last")
    summary_path = os.path.join(args.out_dir, "summary.csv")
    summary.to_csv(summary_path, index=False)
    logger.info(f"Wrote per-run summary -> {summary_path}")

    # --- per-seed table for the appendix ---
    metric_cols = [c for c in summary.columns
                   if c.startswith(("test/graph_accuracy/", "test/node_accuracy/",
                                    "val/graph_accuracy/", "val/node_accuracy/"))]
    if metric_cols:
        table = summary[["seed"] + sorted(metric_cols)]
        md_path = os.path.join(args.out_dir, "per_seed_table.md")
        with open(md_path, "w") as f:
            f.write(table.to_markdown(index=False, floatfmt=".4f"))
        print("\n" + table.to_string(index=False))
        logger.info(f"Wrote per-seed table -> {md_path}")
    else:
        logger.warning("No test/val accuracy columns in the run summaries; "
                       "per-seed table skipped.")

    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump({"entity": args.entity, "project": args.project,
                   "group": args.group, "n_runs": len(runs)}, f, indent=2)


if __name__ == "__main__":
    main()
