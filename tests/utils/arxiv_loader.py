import os
import random
import networkx as nx
import numpy as np
from ogb.nodeproppred import NodePropPredDataset
from loguru import logger

_ARXIV_GRAPH = None

def get_arxiv_graph(root="data/"):
    """
    Lazily loads the OGBN-Arxiv dataset, converts it to an undirected
    NetworkX graph, and caches it to prevent multiple loads.
    """
    global _ARXIV_GRAPH
    if _ARXIV_GRAPH is None:
        logger.info("Loading OGBN-Arxiv dataset...")
        # Will automatically download and load using OGB
        dataset = NodePropPredDataset(name='ogbn-arxiv', root=root)
        graph_data = dataset[0][0]
        
        # graph_data['edge_index'] is 2 x num_edges array
        edges = list(map(tuple, graph_data['edge_index'].T))
        
        logger.info("Converting to an undirected NetworkX graph...")
        G = nx.DiGraph()
        G.add_edges_from(edges)
        
        _ARXIV_GRAPH = G.to_undirected()
        
        # Remove self loops as it's common practice for clean adjacency matrices
        _ARXIV_GRAPH.remove_edges_from(nx.selfloop_edges(_ARXIV_GRAPH))
        
        logger.info(f"Loaded Arxiv graph: {_ARXIV_GRAPH.number_of_nodes()} nodes, {_ARXIV_GRAPH.number_of_edges()} edges")
        
    return _ARXIV_GRAPH


def arxiv_graph_generator(n, seed=None):
    """
    Generates a connected subgraph of exactly `n` nodes from the arxiv dataset.
    Uses randomized BFS expansion starting from a random seed node.
    """
    G = get_arxiv_graph()
    rng = random.Random(seed) if seed is not None else random.Random()
    
    nodes = list(G.nodes())
    
    while True:
        start_node = rng.choice(nodes)
        
        # BFS to collect exactly n nodes
        visited = {start_node}
        queue = [start_node]
        
        while queue and len(visited) < n:
            current = queue.pop(0)
            neighbors = list(G.neighbors(current))
            rng.shuffle(neighbors) # Shuffle to get random traversal/subtrees
            
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
                    if len(visited) == n:
                        break
                        
        if len(visited) == n:
            subgraph = G.subgraph(visited).copy()
            # Double check connectivity, though BFS ensures it.
            if nx.is_connected(subgraph):
                break
                
    # Relabel nodes to 0 ... n-1
    subgraph = nx.convert_node_labels_to_integers(subgraph)
    
    # Return as dense boolean numpy array (float expected by sampler)
    return nx.to_numpy_array(subgraph).astype(float)
