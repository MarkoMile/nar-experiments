# Rebuttal draft — LoG 2026 Extended Abstract

> Numbers below are measured; see RESULTS.md for the full tables and the raw logs
> in this directory. All evaluations are of frozen checkpoints — nothing retrained.

---

## Response to Reviewer 1

We thank the reviewer for a very precise critique. Acting on weakness #2 led us
to a methodological error in our own evaluation that materially changes Table 1, and
we report it here in full, including the parts that do not favour us.

### A. Our reported numbers were produced at an unstated evaluation precision

Table 1 was produced with PyTorch Lightning `precision="16-mixed"` and
`torch.set_float32_matmul_precision('medium')`. Neither was stated in the paper. Both
reduce arithmetic precision at inference; the model weights are fp32 and are unchanged.
Every number in this response is an evaluation of a frozen checkpoint — nothing was
retrained. Re-evaluating our best frozen checkpoint on a fixed set of 200 graphs per bin,
varying only these two settings between rows:

| effective precision | delaunay_800 | delaunay_1600 | er_800 | er_1600 | ws_800 | ws_1600 |
|---|---|---|---|---|---|---|
| fp16 (submitted precision settings) | 0.995 | 0.925 | 1.000 | 0.960 | 0.905 | **0.010** |
| fp32, bf16 matmuls           | 1.000 | 0.980 | 1.000 | 0.995 | 0.990 | **0.495** |
| fp32, fp32 matmuls           | 1.000 | **1.000** | 1.000 | 0.995 | 0.995 | **0.980** |

Accuracy is monotone in arithmetic precision in every bin. At true fp32, node accuracy
is 1.0000 on all bins. WS-1600 moves by a factor of 98; ER-1600 by 1.04.

One discrepancy we disclose proactively: the fp16 row does not reproduce every cell of
the submitted Table 1 — most visibly ws_800, reported as 20.0% (3/15) and measured here
at 0.905 under matched precision settings. We could not isolate the remaining factor
within the rebuttal window; the submission evaluated all 15 graphs of a bin in a single
batch, and while batch size provably does not matter at fp32 (sizes 1/5/15 give
bit-identical results), we have not tested its interaction with fp16; alternatively, the
submitted table may have been produced by a sibling checkpoint of the same training
configuration. We therefore treat
only the fully specified re-runs reported here as authoritative, and will publish the
exact evaluation configuration alongside the revised Table 1.

The ladder is not particular to this checkpoint. Across the four grokked seeds for which
all three rungs were measured, the grokked-mean accuracy climbs the same way:
delaunay_1600 goes 0.914 → 0.978 → 0.995 and ws_1600 goes 0.059 → 0.256 → 0.399. Every
per-seed, per-bin movement is monotone up to sampling noise (largest counter-movement
−0.010 at 200 graphs/bin).

**Corrected Table 1** — our best checkpoint, true fp32, 200 graphs/bin, strict graph
accuracy. Relative to the submitted Table 1: the two precision settings above are
corrected, the sample count is raised from 15 to 200 graphs/bin, and the evaluation batch
size is 5 (the submission used 15). Per-seed results and the spread
across grokked runs are in section B; this checkpoint is at the favourable end of that
spread and should not be read as typical.

| N | ER | Delaunay | WS |
|---|---|---|---|
| 16   | 1.000 | 1.000 | 1.000 |
| 80   | 1.000 | 1.000 | 1.000 |
| 800  | 1.000 | 1.000 | 0.995 |
| 1600 | 0.995 | 1.000 | 0.980 |

A 200/200 cell carries a 95% Clopper–Pearson lower bound of 0.982; 196/200 (the ws_1600
cell) gives [0.950, 0.995]. We will state the evaluation precision in the paper and
report at true fp32.

### B. Weakness #1 — topology dependence

The reviewer is right that a topology-dependent gap exists, and we retain that
conclusion. But we would like to address two parts of the argument.

*The magnitude was an artefact.* WS-1600 was reported as 0.0%. At true fp32 the same
checkpoint scores 0.980 with node accuracy 1.0000. The "persistently high node accuracy
with 0% graph accuracy" the reviewer identifies is the signature of fp16 rounding
accumulating over the rollout, not of a learned heuristic.

*The gap is real across seeds, and smaller than reported.* Averaged over the five
grokked seeds at true fp32 (200 graphs/bin; n = 5 seeds, so the stds are coarse):

| bin | mean ± std | range |
|---|---|---|
| delaunay_800  | 0.991 ± 0.020 | 0.955 – 1.000 |
| delaunay_1600 | 0.955 ± 0.090 | 0.795 – 1.000 |
| er_1600       | 0.813 ± 0.266 | 0.430 – 1.000 |
| ws_1600       | 0.319 ± 0.452 | 0.000 – 1.000 |

So we concede the substance of the reviewer's point: performance is topology-dependent
(Delaunay > ER > WS) and this is not removed by fixing precision. We also note that our
best checkpoint (WS-1600 = 0.980) sits near the top of the grokked range and is not
representative — a fact we would not have reported without this analysis.

What we do question is the inference "distributed small errors ⇒ heuristic, not
algorithm." A model executing a heuristic would likely not reach 1.000 strict graph accuracy on
1600-node Delaunay graphs (BFS depth ≈ 22) with node accuracy 1.0000 once its arithmetic
is not rounded.

*We tested and rejected five explanations for the WS deficit*, and report them so they
are not repeated:
- **BFS-parent tie-breaking**: WS has the *fewest* ambiguous parents (43.4% vs Delaunay
  56.2%, ER 62.1%) yet performs worst — anti-correlated.
- **Small-world shortcuts** (our own explanation in the submission, lines 62–64):
  a controlled sweep over the WS rewiring probability at fixed degree (N=800, k=6, true
  fp32) gives **1.000 at every p ∈ {0.01, 0.05, 0.1, 0.2, 0.5, 1.0}**. The only imperfect
  bin is p=0, the pure ring lattice (0.900), which is also the *deepest* (BFS depth 134).
  Note p=1 fully rewires the graph — no planar, geometric or lattice regularity remains —
  and accuracy there is 1.000, which speaks directly to the concern that our results
  exploit planar/low-degree regularity.
  Error rate on nodes whose true BFS parent is a rewired shortcut: 0.000%. **Our stated
  explanation is unsupported and we will remove it.**
- **Low degree**: sweeping k ∈ {4,6,8} × p ∈ {0.05,0.1,0.2} at N=1600 gives 0.933–1.000.
- **Batched over-rollout**: test batch size 1 / 5 / 15 gives bit-identical results
  at fp32 (untested at fp16 — see the disclosure in section A).
- **Fuzzy autoregressive mask feedback**: discretising the re-injected `reach_h` changes
  nothing; the mask is already saturated (0% of values in (0.02, 0.98)).

The one measurement that survives is descriptive, not mechanistic: WS has by far the
thinnest tail of pointer decision margins (5th percentile 0.56, vs 27.6 for Delaunay and
11.4 for ER), which is consistent with it being the family most sensitive to any
numerical perturbation. We do not claim a mechanism beyond that.

### B2. Path graphs: long-rollout evidence against the heuristic reading

A path graph is the strongest available test of the reviewer's hypothesis: degree 2,
zero clustering, no shortcuts, no planarity, and nothing a topology-exploiting heuristic
could use. It also forces rollouts far longer than anything in the submission. Evaluated
at true fp32 on the same frozen checkpoint (15 graphs per configuration):

| configuration | strict graph accuracy | node accuracy | mean BFS depth (rollout steps) |
|---|---|---|---|
| path_d512  | 15/15 (**100%**) | 1.0000 | 375 |
| path_d800  | 15/15 (**100%**) | 1.0000 | 606 |
| path_d1200 | 5/5 (**100%**)   | 1.0000 | **800** (max ~1150) |

In our earlier fp16 evaluation of this configuration, strict accuracy at path_d800
was **0.00**. The model executes BFS exactly for 800 sequential steps on average (single
graphs reach ~1150) on a topology where no heuristic exists, and the reported failure is
again arithmetic rather than algorithmic.

### C. Weakness #2 — statistical rigor

Accepted, and now addressed.

*How runs were selected.* Grokked runs were identified by inspecting validation curves
(validation accuracy and training weight norm — the two panels of Figure 1); test
performance played no role. In this population that inspection involves no judgement
call, because the runs are cleanly bimodal on both signals: maximum `train/weight_norm`
reaches **196.2–268.3** for the five grokked runs versus **55.8–160.4** for the other
five, and validation graph accuracy on delaunay_800 is ≥0.933 versus ≤0.400. Any
threshold anywhere inside either gap reproduces the same 5/10 split, so the selection
could not have come out differently.

*Variance.* Per-seed results for all ten runs, plus mean ± std over the grokked runs, are
in the table above and will be added as an appendix table. We report the grokked subset
because the claim is that this training regime *can* reach the algorithmic solution, and
averaging over a bimodal population describes no run that exists — but we state the 5/10
rate prominently and give every seed individually.

*Sample size.* The submitted Table 1 used 15 graphs per bin. All numbers here use 200.

### D. Weakness #3 — single-seed ablations

Accepted without qualification. Multiseed ablations require retraining, which we could
not afford in the rebuttal window, due to compute limitations. We will state this as a limitation.

### E. Weakness #4 — overclaiming from one task

Accepted. We will soften the abstract and conclusion: our evidence supports a claim about
BFS under this training regime, not about "unlocking latent algorithmic capabilities in
standard graph architectures" in general.

All raw evaluation logs behind the numbers above are retained, and we are happy to run
any further check the reviewer considers decisive during the discussion period.

---

## Response to Reviewer 2

Thank you — your question about ER small-worldness prompted a check that corrected a
genuine error in our paper.

**On ER edge probability.** Your reasoning holds for *fixed* p, but SALSA-CLRS scales p
as ~1/N to hold mean degree constant: er_800 uses p ∈ [0.008, 0.025] and er_1600 uses
p ∈ [0.004, 0.0125], both giving mean degree ≈ 6.4–20 independent of N. So diameter grows
as ~log N / log c rather than staying at 2; we measure BFS depth ≈ 4.6 at N=1600. The
"predict 1 if adjacent, else 2" shortcut is therefore not available.

**But your underlying instinct is correct, and it improves the paper.** ER is by far the
shallowest of the three families (BFS depth 4.6 at N=1600, versus 11.5 for WS and 22.4 for
Delaunay). That makes ER the *least* demanding of our evaluations, and it means the
Delaunay result — the deepest topology, where we now measure 0.955 ± 0.090 across grokked
seeds at N=1600 — is the meaningful one, not the ER number.

**Our WS explanation was wrong.** You identified our claim that "small-world shortcuts
heavily penalize fuzzy heuristics" as unintuitive. It is worse than unintuitive: a
controlled sweep over the WS rewiring probability at fixed degree gives 1.000 strict
accuracy at *every* nonzero rewiring probability, including p=1. Adding shortcuts does not
hurt. We are removing that sentence and replacing it with the measured facts rather than a
mechanism we cannot support.

We have also corrected an unstated evaluation-precision issue that substantially changes
Table 1 (see our response to Reviewer 1); WS-1600 in particular was reported as 0.0% and
is 0.980 at true fp32 for our best checkpoint.
