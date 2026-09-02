# LoG 2026 rebuttal — measured results

All evaluations are of **frozen checkpoints**. No retraining. Unless stated,
200 graphs per bin (the paper's Table 1 used 15).

## 1. Evaluation precision was never stated, and it dominates the WS result

Two settings in the repo's own eval scripts control arithmetic precision:
`TRAIN.PRECISION` (Lightning autocast) and `torch.set_float32_matmul_precision`
(`eval_checkpoint.py`, `eval_path.py`, `eval_directed.py`, `test_obgn_arxiv_large.py`,
`test_sbm.py` all set `'medium'`; `bfs_depth_analysis.py` sets none, i.e. `'highest'`).

`model-best.ckpt`, 200 graphs/bin, graph accuracy:

| setting                          | delaunay_800 | delaunay_1600 | er_800 | er_1600 | ws_800 | ws_1600 |
|----------------------------------|--------------|---------------|--------|---------|--------|---------|
| 16-mixed + medium  (**the paper**) | 0.995 | 0.925 | 1.000 | 0.960 | 0.905 | **0.010** |
| 16-mixed + highest               | 0.995 | 0.925 | 1.000 | 0.960 | 0.905 | **0.010** |
| fp32 + medium                    | 1.000 | 0.980 | 1.000 | 0.995 | 0.990 | **0.495** |
| fp32 + highest (**true fp32**)   | 1.000 | **1.000** | 1.000 | 0.995 | 0.995 | **0.980** |

With `16-mixed`, the matmul setting is irrelevant (autocast puts matmuls in fp16
regardless) — rows 1 and 2 are identical. So this collapses to ONE ordered axis of
effective precision: fp16 -> fp32-with-bf16-matmuls -> true fp32, giving
WS-1600 **0.010 -> 0.495 -> 0.980**. Every bin improves monotonically.
At true fp32, node accuracy is 1.0000 on every bin.

## 2. Across seeds the picture is much weaker — model-best is a favourable checkpoint

True fp32 (`precision=32, matmul=highest`), 200 graphs/bin — ALL 10 SEEDS:

| seed | grok | delaunay_800 | delaunay_1600 | er_800 | er_1600 | ws_800 | ws_1600 |
|---|---|---|---|---|---|---|---|
| 42 | Y | 1.000 | 0.985 | 0.920 | 0.635 | 0.300 | 0.000 |
| 44 | Y | 1.000 | 0.995 | 1.000 | 1.000 | 0.815 | 0.025 |
| 46 | Y | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 50 | Y | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.570 |
| 51 | Y | 0.955 | 0.795 | 0.770 | 0.430 | 0.075 | 0.000 |
| 43 | n | 0.185 | 0.010 | 0.015 | 0.000 | 0.000 | 0.000 |
| 45 | n | 0.400 | 0.140 | 0.010 | 0.000 | 0.000 | 0.000 |
| 47 | n | 0.415 | 0.075 | 0.160 | 0.020 | 0.000 | 0.000 |
| 48 | n | 0.495 | 0.180 | 0.250 | 0.045 | 0.000 | 0.000 |
| 49 | n | 0.285 | 0.085 | 0.010 | 0.000 | 0.000 | 0.000 |
| **mean (grok)** | | **0.991** | **0.955** | **0.938** | **0.813** | **0.638** | **0.319** |
| **std (grok, n=5)** | | 0.020 | 0.090 | 0.100 | 0.266 | 0.426 | 0.452 |
| mean (not grok) | | 0.356 | 0.098 | 0.089 | 0.013 | 0.000 | 0.000 |

`model-best` (ws_1600 = 0.980) sits near the TOP of the grokked range and is NOT
representative. Topology dependence is real and survives the precision fix:
Delaunay 0.955 > ER 0.813 > WS 0.319 (grokked mean).

**Precision effect concentrates in near-exact runs.** Comparing fp32+medium vs
fp32+highest per seed: non-grokked seeds change by at most 0.025 in any bin, while
grokked seeds gain up to +0.285 (seed 46, ws_1600). Caveat: non-grokked seeds sit at
0.000 on the sensitive bins, so a floor effect contributes — this shows the effect is
concentrated in runs that are already near-exact, not that grokking causes it.

## 3. Grokking split is objective and training-time

Max `train/weight_norm` over training (from W&B, group `multiseed`):

| | max weight_norm | max val/node_accuracy/ws_800 |
|---|---|---|
| grokked (42,44,46,50,51)     | **196.2 - 268.3** | 0.9947 - 1.0000 |
| not grokked (43,45,47,48,49) | **55.8 - 160.4**  | 0.9617 - 0.9928 |

Any threshold in 160-196 reproduces the paper's 5/10 exactly, using a training
signal (not test data). Validation `graph_accuracy/delaunay_800` separates the same
way (grokked >= 0.933, not grokked <= 0.400). The split needs no judgement call.
At true fp32 the test split is also clean: grokked delaunay_800 >= 0.955 vs
not grokked <= 0.415.

## 4. Hypotheses tested and REJECTED (do not put these in the rebuttal)

- **BFS-parent tie-breaking.** WS has the FEWEST ambiguous parents (43.4% vs
  Delaunay 56.2%, ER 62.1% at N=1600) yet fails most. Anti-correlated.
- **Small-world shortcuts.** Controlled WS sweep (N=800, k=6, fixed p, true fp32,
  30 graphs/bin): graph accuracy 1.000 at every p in {0.01,0.05,0.1,0.2,0.5,1.0}.
  The ONLY imperfect bin is p=0 (pure ring lattice, 0.900) — which is the DEEPEST
  (BFS depth 134). Shortcut-parent node error rate: 0.000% throughout.
  The paper's stated WS explanation is unsupported.
- **Low degree k.** N=1600 sweep over k in {4,6,8} x p in {0.05,0.1,0.2} at true
  fp32: every bin 0.933-1.000 (mean 0.977).
- **fp16 rounding at the output decode.** 0% of final pointer margins fall below
  fp16 ULP. The damage accumulates through the rollout, not at the decode.
- **Batched over-rollout** (`max_len = batch.length.max()`). Test batch size 1 vs
  5 vs 15 gives bit-identical results in every bin.
- **Fuzzy mask feedback.** Discretising the re-injected `reach_h` at 0.5 changes
  nothing at either precision. Measured cause: the mask is already saturated —
  0% of its values lie in (0.02, 0.98).

## 5. The one mechanistic thread that survived

Pointer decision margin (top1 - top2 logit per node), N=800, true fp32:

| family | median | p05 | p01 | min |
|---|---|---|---|---|
| ws       | 485.0 | **0.561** | 0.459 | 0.330 |
| delaunay | 349.3 | **27.6**  | 7.20  | 0.848 |
| er       | 181.2 | 11.4      | 1.69  | 0.585 |

WS has the highest median but a tail ~49x thinner than Delaunay's. That explains
WHICH topology is most fragile to numerical perturbation. It does NOT establish the
magnitude or path of the perturbation — do not claim a mechanism.

## 6. Structural reference numbers (CPU, networkx)

| N=1600 | mean deg | BFS depth | clustering | ambiguous-parent nodes |
|---|---|---|---|---|
| delaunay | 6.0 | 22.4 | 0.437 | 56.2% |
| ws       | 6.0 | 11.5 | 0.400 | 43.4% |
| er       | 13.6 | 4.6 | 0.009 | 62.1% |

SALSA-CLRS scales ER `p` as ~1/N (er_800 p_range [0.008,0.025], er_1600
[0.004,0.0125]) so mean degree is constant 6.4-20 across sizes. R2's premise that
ER graphs have diameter ~2 assumes FIXED p and does not hold here; diameter grows
as ~log N / log c. But R2's underlying instinct is right: ER is by far the
shallowest of the three families.

## 7. Path graphs — measured at TRUE fp32 (precision=32, matmul=highest)

Degree 2, zero clustering, no shortcuts, no planarity: nothing for a topology-exploiting
heuristic to use. Rollout length is max(s, n-1-s) for a uniformly random source.

| config | correct | node accuracy | mean BFS depth (rollout steps) |
|---|---|---|---|
| path_d512  | 15/15 (100%) | 1.0000 | 375.1 (std 82.5) |
| path_d800  | 15/15 (100%) | 1.0000 | 606.5 (std 107.6) |
| path_d1200 | 5/5 (100%)   | 1.0000 | 800.4 (std 97.3) |

Under the submitted settings (fp16), the same checkpoint scores 0.00 at path_d800.
This is a third independent instance of the precision effect, on a topology unrelated
to WS. NOTE: the "1153 steps" figure in media/ was a single graph whose source happened
to sit near an end; the mean for that configuration is 800.

## 7b. Path graphs (earlier runs, media/, at the historical 'medium' default)

fp32: 100% strict graph accuracy at depths 16, 32, 64, 128, 256, 512 (15 samples
each) and at d1200 (**1153 sequential BFS steps**, n=1 sample only).
fp16: collapses — d160 -> 0.30, d800 -> 0.00.
Degree-2 graphs, zero clustering, no shortcuts. Needs a rerun at 15 samples for d1200.
