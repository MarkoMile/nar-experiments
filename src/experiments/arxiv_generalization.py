# This script tests the generalization capabilities of the trained model on the ogbn-arxiv dataset.
# It should include:
# - Loading the pre-trained SALSA model from a checkpoint.
# - Loading the ogbn-arxiv dataset using src/utils/arxiv_loader.py.
# - Logic to sample random subgraphs from the arxiv citation graph.
# - Running the BFS task on these subgraphs.
# - Evaluating the model's performance on this out-of-distribution data.
# - Comparison metrics between training distribution and the arxiv subgraphs.
