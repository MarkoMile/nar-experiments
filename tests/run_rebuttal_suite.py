"""
Driver for the LoG 2026 rebuttal experiment suite.

Runs every evaluation the rebuttal depends on, in the order that the rebuttal
text blocks on, so that a run which dies partway still leaves the load-bearing
results on disk. Nothing here trains: every stage evaluates frozen checkpoints.

Stages
    smoke     Tiny N=1600 run to catch OOM / config breakage before the long jobs.
    depth     Mistake-depth analysis per checkpoint, both precisions. This is the
              evidence that Delaunay/ER errors are single-node and terminal while
              WS errors are distributed and mid-rollout.
    table1    Full Table 1 re-run per checkpoint, both precisions, high sample
              count. Table 1 in the paper used 15 graphs per bin at 16-mixed.
    wssweep   WS rewiring sweep (fixed k, p is the only variable) with shortcut
              error localisation. Isolates whether shortcuts cause WS failures.
    disc      Soft vs hard re-injected mask hint on WS, inference only.
    path      Path graphs in fp32 up to depth 1200 (~1153 sequential BFS steps).

Usage:
    python tests/run_rebuttal_suite.py --ckpts model-checkpoints/model-best.ckpt \
        --out-dir results/rebuttal --num-samples 200
"""

import os
import sys
import glob
import time
import json
import argparse
import subprocess

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

ALL_STAGES = ["smoke", "depth", "table1", "wssweep", "disc", "path"]

# Rewiring probabilities for the WS sweep. p=0 is a pure ring lattice (no
# shortcuts at all); the paper's ws bins sample p in [0.05, 0.2].
WS_P_SWEEP = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]

# Path depths. 1200 is the one that matters (~1153 sequential rollout steps);
# the smaller ones give the curve showing where fp16 breaks down and fp32 does not.
PATH_DEPTHS = [16, 32, 64, 128, 256, 512, 800, 1200]


class Runner:
    def __init__(self, out_dir, dry_run=False, stop_on_fail=False):
        self.out_dir = out_dir
        self.dry_run = dry_run
        self.stop_on_fail = stop_on_fail
        self.log_dir = os.path.join(out_dir, "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        self.results = []

    def run(self, name, argv):
        cmd = [sys.executable, "-u"] + argv
        printable = " ".join(cmd)
        print(f"\n{'=' * 78}\n▶ {name}\n  {printable}\n{'=' * 78}", flush=True)
        if self.dry_run:
            self.results.append({"name": name, "rc": None, "secs": 0.0, "cmd": printable})
            return 0

        log_path = os.path.join(self.log_dir, f"{name}.log")
        start = time.time()
        with open(log_path, "w") as log:
            log.write(printable + "\n\n")
            log.flush()
            proc = subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in proc.stdout:
                sys.stdout.write(line)
                log.write(line)
            rc = proc.wait()
        secs = time.time() - start

        status = "ok" if rc == 0 else f"FAILED (rc={rc})"
        print(f"◀ {name}: {status} in {secs / 60:.1f} min  (log: {log_path})", flush=True)
        self.results.append({"name": name, "rc": rc, "secs": secs, "cmd": printable})
        if rc != 0 and self.stop_on_fail:
            self.summary()
            sys.exit(f"Stopping: {name} failed and --stop-on-fail is set.")
        return rc

    def summary(self):
        print(f"\n{'=' * 78}\nRUN SUMMARY\n{'=' * 78}")
        print(f"{'job':<46}{'status':>10}{'minutes':>10}")
        print("-" * 78)
        for r in self.results:
            status = "dry-run" if r["rc"] is None else ("ok" if r["rc"] == 0 else f"rc={r['rc']}")
            print(f"{r['name']:<46}{status:>10}{r['secs'] / 60:>10.1f}")
        path = os.path.join(self.out_dir, "run_summary.json")
        with open(path, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"\nWrote {path}")
        failed = [r["name"] for r in self.results if r["rc"] not in (0, None)]
        if failed:
            print(f"\nFAILED JOBS: {', '.join(failed)}")


def ckpt_tag(path):
    return os.path.splitext(os.path.basename(path))[0]


def main():
    p = argparse.ArgumentParser(description="Run the rebuttal experiment suite")
    p.add_argument("--ckpts", type=str, nargs="+", required=True,
                   help="Checkpoint paths (globs allowed)")
    p.add_argument("--out-dir", type=str, default="results/rebuttal")
    p.add_argument("--stages", type=str, nargs="+", default=ALL_STAGES, choices=ALL_STAGES)
    p.add_argument("--num-samples", type=int, default=200,
                   help="Graphs per bin for depth/table1. Paper used 15.")
    p.add_argument("--depth-samples", type=int, default=None,
                   help="Override graphs per bin for the depth stage (defaults to --num-samples).")
    p.add_argument("--sweep-samples", type=int, default=50, help="Graphs per WS sweep bin")
    p.add_argument("--path-samples", type=int, default=15, help="Graphs per path depth")
    p.add_argument("--precisions", type=str, nargs="+", default=["32", "16-mixed"])
    p.add_argument("--sizes", type=int, nargs="+", default=[800, 1600])
    p.add_argument("--batch-size", type=int, default=1, help="bfs_depth_analysis batch size")
    p.add_argument("--test-batch-size", type=int, default=5, help="eval_checkpoint test batch size")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-cores", type=int, default=-1,
                   help="Cores for graph generation (-1 = serial). Generation is serial by "
                        "default and dominates wall-clock for high sample counts; set this to "
                        "the instance's core count.")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--stop-on-fail", action="store_true")
    args = p.parse_args()

    ckpts = []
    for pattern in args.ckpts:
        hits = sorted(glob.glob(pattern))
        if not hits:
            sys.exit(f"No checkpoint matched {pattern!r}")
        ckpts.extend(hits)
    print(f"Checkpoints ({len(ckpts)}):")
    for c in ckpts:
        print(f"  {c}")

    depth_samples = args.depth_samples or args.num_samples
    out = args.out_dir
    os.makedirs(out, exist_ok=True)
    r = Runner(out, dry_run=args.dry_run, stop_on_fail=args.stop_on_fail)

    DEPTH = "tests/bfs_depth_analysis.py"
    EVAL = "tests/eval_checkpoint.py"
    PATH = "tests/eval_path.py"

    # --- smoke: fail fast on OOM / config breakage -------------------------
    if "smoke" in args.stages:
        r.run("smoke", [DEPTH, "--ckpt", ckpts[0], "--sizes", "1600",
                        "--families", "ws", "er", "delaunay",
                        "--num-samples", "2", "--batch-size", str(args.batch_size),
                        "--max-cores", str(args.max_cores), "--seed", str(args.seed)])

    # --- depth: the mistake-depth evidence ---------------------------------
    if "depth" in args.stages:
        for c in ckpts:
            for prec in args.precisions:
                tag = f"depth_{ckpt_tag(c)}_p{prec.replace('-', '')}"
                r.run(tag, [DEPTH, "--ckpt", c,
                            "--sizes", *[str(s) for s in args.sizes],
                            "--families", "ws", "er", "delaunay",
                            "--num-samples", str(depth_samples),
                            "--batch-size", str(args.batch_size),
                            "--precision", prec, "--seed", str(args.seed),
                            "--max-cores", str(args.max_cores),
                            "--json-out", os.path.join(out, f"{tag}.json")])

    # --- table1: the statistics R1 asked for -------------------------------
    if "table1" in args.stages:
        for c in ckpts:
            for prec in args.precisions:
                tag = f"table1_{ckpt_tag(c)}_p{prec.replace('-', '')}"
                r.run(tag, [EVAL, "--ckpt", c, "--num-samples", str(args.num_samples),
                            "--test-batch-size", str(args.test_batch_size),
                            "--precision", prec, "--seed", str(args.seed),
                            "--max-cores", str(args.max_cores),
                            "--num-workers", "4"])

    # --- wssweep: does p alone explain the WS failure? ---------------------
    if "wssweep" in args.stages:
        for c in ckpts:
            tag = f"wssweep_{ckpt_tag(c)}"
            r.run(tag, [DEPTH, "--ckpt", c, "--sizes", "800",
                        "--families",  # deliberately empty: sweep bins only
                        "--ws-p-sweep", *[str(v) for v in WS_P_SWEEP],
                        "--ws-k", "6", "--shortcuts",
                        "--num-samples", str(args.sweep_samples),
                        "--batch-size", str(args.batch_size),
                        "--precision", "32", "--seed", str(args.seed),
                        "--max-cores", str(args.max_cores),
                        "--json-out", os.path.join(out, f"{tag}.json")])

    # --- disc: soft vs hard mask feedback ----------------------------------
    if "disc" in args.stages:
        for c in ckpts:
            for mode in ("soft", "hard"):
                tag = f"disc_{ckpt_tag(c)}_{mode}"
                r.run(tag, [DEPTH, "--ckpt", c, "--sizes", *[str(s) for s in args.sizes],
                            "--families", "ws",
                            "--num-samples", str(args.sweep_samples),
                            "--batch-size", str(args.batch_size),
                            "--precision", "32", "--mask-mode", mode,
                            "--seed", str(args.seed), "--max-cores", str(args.max_cores),
                            "--json-out", os.path.join(out, f"{tag}.json")])

    # --- path: the anti-heuristic result -----------------------------------
    if "path" in args.stages:
        for c in ckpts:
            tag = f"path_{ckpt_tag(c)}"
            r.run(tag, [PATH, "--ckpt", c,
                        "--max-depths", *[str(d) for d in PATH_DEPTHS],
                        "--num-samples", str(args.path_samples),
                        "--precision", "32", "--seed", str(args.seed)])

    r.summary()


if __name__ == "__main__":
    main()
