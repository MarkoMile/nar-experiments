import networkx as nx
import numpy as np

def sbm_graph_generator(n, seed=None):
    """
    Generates a Stochastic Block Model (SBM) graph and returns its dense numpy adjacency matrix.
    
    Dynamically constructs 2 to 5 communities and assigns dense intra-community connectivity
    and sparse inter-community connectivity.
    """
    if seed is not None:
        np.random.seed(seed)
        
    num_blocks = np.random.randint(2, 6) # 2 to 5 communities
    # Randomly distribute nodes roughly equally
    sizes = np.random.multinomial(n, np.ones(num_blocks)/num_blocks)
    
    # Edge case: prevent size 0 blocks
    while min(sizes) == 0 and len(sizes) > 1:
        zero_idx = np.argmin(sizes)
        max_idx = np.argmax(sizes)
        sizes[max_idx] -= 1
        sizes[zero_idx] += 1
            
    # Probabilities:
    # Intra-community (main diagonal): 0.15 to 0.5
    # Inter-community (off-diagonal): 0.01 to 0.08
    p = np.zeros((num_blocks, num_blocks))
    for i in range(num_blocks):
        for j in range(num_blocks):
            if i == j:
                p[i, j] = np.random.uniform(0.15, 0.5)
            elif i < j:
                prob = np.random.uniform(0.01, 0.08)
                p[i, j] = prob
                p[j, i] = prob  # Ensure symmetry for undirected graphs
                
    # Generate undirected graph
    G = nx.stochastic_block_model(sizes, p, directed=False, seed=seed)
    
    # We optionally can just return this dense array, which salsa-clrs handles natively
    adj = nx.to_numpy_array(G)
    
    # Binarize in case weights leaked in
    adj = (adj > 0).astype(float)
    return adj
