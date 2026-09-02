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
