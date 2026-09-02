# NAR Experiments

Neural Algorithmic Reasoning (NAR) experiments on graph algorithms (currently BFS), training on SALSA-CLRS-generated graphs and evaluating out-of-distribution generalization to larger, structurally different held-out graphs.

## Language

**Graph accuracy**:
The fraction of test graphs where every node's output matches the ground truth exactly (exact whole-graph match). The strict, all-or-nothing metric — this is what "80% accuracy on N=1600 Delaunay" refers to.
_Avoid_: Accuracy (ambiguous with node accuracy), graph F1 (identical to graph accuracy in this codebase, no longer logged separately).

**Node accuracy**:
The fraction of individual node outputs (across all test graphs) that match ground truth. Far more forgiving than graph accuracy — a graph with one wrong node still scores near-1.0 node accuracy despite scoring 0 on graph accuracy.
_Avoid_: Accuracy (ambiguous with graph accuracy).

**Rewiring fraction (p)**:
The Watts-Strogatz-style per-edge rewiring probability: for each edge in a base graph, with probability p, one endpoint is reassigned to a random other node (avoiding self-loops/duplicate edges, retrying to preserve connectivity). p=0 leaves the base graph unchanged; p=1 rewires every edge, tending toward a random-like graph regardless of the base topology. This is the existing mechanism behind the `ws` (Watts-Strogatz) generator, applied here to a different base structure.

**Rewired-Delaunay graph**:
A Delaunay triangulation graph with a fraction p of its edges rewired to random long-range links (see Rewiring fraction). Used to find where the model's generalization breaks down as local geometric structure is replaced by random long-range connections. At p=0 it is exactly a Delaunay graph; as p→1 it approaches a random (ER-like) graph of the same edge count — measured clustering falls 0.44 → 0.006 and diameter 20 → 7 at N=800. Note it does *not* approach the `ws` family: Watts-Strogatz graphs in this project use low p and so stay highly clustered (~0.44), structurally near this axis's p=0 end, not its p=1 end.
_Avoid_: Perturbed Delaunay, noisy Delaunay.

**Grokked seed**:
A training run whose maximum `train/weight_norm` exceeds ~180 (measured population: grokked runs reach 196–268, non-grokked stay at 56–160; any threshold in 160–196 gives the same 5/10 split). Defined on training dynamics only — never on test metrics, which would make conditional reporting circular. Validation graph accuracy on delaunay_800 separates the same runs independently (≥0.933 vs ≤0.400).
_Avoid_: "successful run" (vague), classifying by eye from figures (not reproducible), any test-set-based criterion.

**Effective evaluation precision**:
The arithmetic precision a frozen model is actually evaluated at — set jointly by the Lightning `precision` flag and `torch.set_float32_matmul_precision`. Three rungs occur in this repo: fp16 autocast (matmul setting irrelevant), fp32 with bf16 tensor-core matmuls (`'medium'`, the historical hardcoded default in every eval script except `bfs_depth_analysis.py`), and true fp32 (`'highest'`). Graph accuracy is monotone in this axis in every measured bin; WS bins are by far the most sensitive (ws_1600 spans 0.01 → 0.98 across the rungs on the best checkpoint). Any reported accuracy is meaningless without stating the rung.
_Avoid_: "fp32" for the `'medium'` setting (it is not true fp32), comparing accuracies across unstated rungs.
