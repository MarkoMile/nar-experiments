This directory stores datasets.

1.  **Synthetic Data**: Generated on-the-fly or cached here by `src/utils/graph_generation.py`.
2.  **ogbn-arxiv**: The Open Graph Benchmark arxiv dataset.
    *   This will be downloaded automatically by the OGB library or `src/utils/arxiv_loader.py`.
    *   Ensure this directory is ignored by git if files become large.
