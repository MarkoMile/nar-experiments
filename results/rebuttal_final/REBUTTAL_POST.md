**We thank both reviewers for the careful reading.** Acting on Reviewer dWqK's weakness #2
led us to a methodological error in our evaluation that materially changes Table 1; we
report it in full below, including the parts that do not favour us. All numbers are
evaluations of frozen checkpoints — nothing was retrained. A revised PDF with the
corrected Table 1 and a per-seed appendix has been uploaded.

# To Reviewer dWqK

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

One discrepancy we disclose proactively: the fp16 row does not reproduce the submitted
ws_800 (reported 20.0% = 3/15; we measure 0.905 at matched precision). We could not
isolate the cause in the window — candidates are the submission's batch size of 15 under
fp16 (batch size is provably irrelevant at fp32: sizes 1/5/15 are bit-identical) or a
sibling checkpoint of the same training configuration — so we treat only the fully
specified re-runs here as authoritative.

The ladder is not particular to this checkpoint: across grokked seeds the mean climbs the
same way (delaunay_1600: 0.914 → 0.978 → 0.995; ws_1600: 0.059 → 0.256 → 0.399), monotone
per seed and per bin up to sampling noise.

**Corrected Table 1** — best checkpoint, true fp32, 200 graphs/bin (submission: 15),
strict graph accuracy. This checkpoint is at the favourable end of the grokked-seed
spread (section B) and should not be read as typical.

| N | ER | Delaunay | WS |
|---|---|---|---|
| 16   | 1.000 | 1.000 | 1.000 |
| 80   | 1.000 | 1.000 | 1.000 |
| 800  | 1.000 | 1.000 | 0.995 |
| 1600 | 0.995 | 1.000 | 0.980 |

(A 200/200 cell has a 95% Clopper–Pearson lower bound of 0.982.) The revised PDF states the evaluation precision and reports at true fp32.

### B. Weakness #1 — topology dependence

The reviewer is right that a topology-dependent gap exists. But we would like to address
two parts of the argument.

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

*We tested and rejected five explanations for the WS deficit* so they are not repeated:
- **Small-world shortcuts** (our own explanation, lines 62–64): a rewiring-probability
  sweep at fixed degree (N=800, k=6, fp32) gives **1.000 at every p ∈ {0.01…1.0}** — and
  p=1 destroys all planar/geometric regularity, which speaks directly to the concern that
  our results exploit it. Only p=0 (ring lattice, the *deepest* bin, BFS depth 134) is
  imperfect at 0.900. **Our stated explanation is unsupported; it is removed in the
  revised PDF.**
- **Tie-breaking**: WS has the *fewest* ambiguous BFS parents (43.4% vs 56.2/62.1%) yet
  performs worst. **Low degree**: k ∈ {4,6,8} × p ∈ {0.05–0.2} at N=1600 gives
  0.933–1.000. **Batched over-rollout**: batch sizes 1/5/15 are bit-identical at fp32.
  **Fuzzy mask feedback**: discretising the re-injected hint changes nothing (it is
  already saturated).

What survives is descriptive, not mechanistic: WS has by far the thinnest tail of pointer
decision margins (5th percentile 0.56 vs 27.6 Delaunay, 11.4 ER) — consistent with being
the family most sensitive to numerical perturbation. We claim no mechanism beyond that.

### B2. Path graphs: long-rollout evidence against the heuristic reading

A path graph is the strongest available test of the reviewer's hypothesis: degree 2, zero
clustering, no shortcuts, no planarity — nothing a topology-exploiting heuristic could
use — and far longer rollouts than anything in the submission. Same frozen checkpoint,
true fp32:

| configuration | strict graph accuracy | node accuracy | mean BFS depth (rollout steps) |
|---|---|---|---|
| path_d512  | 15/15 (**100%**) | 1.0000 | 375 |
| path_d800  | 15/15 (**100%**) | 1.0000 | 606 |
| path_d1200 | 5/5 (**100%**)   | 1.0000 | **800** (max ~1150) |

Our earlier fp16 evaluation of this configuration scored **0.00** at path_d800. The model
executes BFS exactly for ~800 sequential steps on a topology with no structural shortcut
(pointers depend on information hundreds of hops away); the failure was arithmetic, not
algorithmic.

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

*Variance.* Per-seed results for all ten runs are in the revised PDF's new appendix. We
report means over the grokked subset because the claim is that this regime *can* reach
the algorithmic solution, and averaging a bimodal population describes no run that exists
— but the 5/10 rate is stated prominently and every seed is given individually.

*Sample size.* The submitted Table 1 used 15 graphs per bin. All numbers here use 200.

### D. Weakness #3 — single-seed ablations

Accepted without qualification. Multiseed ablations require retraining, which compute
limitations put beyond the rebuttal window. We will state this as a limitation.

### E. Weakness #4 — overclaiming from one task

Accepted. We will soften the abstract and conclusion: our evidence supports a claim about
BFS under this training regime, not about "unlocking latent algorithmic capabilities in
standard graph architectures" in general.

All raw logs behind these numbers are retained; we are happy to run any further check the
reviewer considers decisive during the discussion period.

---

# To Reviewer SQyD

Thank you — your ER small-worldness question prompted a check that corrected a genuine
error in our paper.

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
Table 1 (see our response to Reviewer dWqK): WS-1600, reported as 0.0%, is 0.980 at true
fp32 for our best checkpoint.
