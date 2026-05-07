"""
SALSA-CLRS library monkey-patches.

This module contains all runtime patches applied to the salsaclrs library
to support features not natively available (directed graphs, additional graph
generators, sampler registration, etc.).

Importing this module applies all patches as a side effect.
"""

import torch
import networkx as nx
import numpy as np
from salsaclrs.sampler import Sampler, BfsSampler, BellmanFordSampler, SAMPLERS
import salsaclrs.data as custom_data

from src.utils.config import get_cfg_defaults

# ---------------------------------------------------------------------------
# 1. Disable sparseness verification
#    In directed graphs, BFS pointers (pi) point from child to parent, which
#    may not be an edge defined in the original asymmetric `edge_index`.
# ---------------------------------------------------------------------------
def patched_verify_sparseness(data, edge_index, data_name):
    pass  # Disable sparseness checking so backward edges don't crash dataset generation

custom_data.verify_sparseness = patched_verify_sparseness


# ---------------------------------------------------------------------------
# 2. Force torch.load(weights_only=False)
#    salsaclrs uses older pickle-based caching and we don't control the
#    library code to pass weights_only=False down.
# ---------------------------------------------------------------------------
original_torch_load = torch.load

def unsafe_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)

torch.load = unsafe_torch_load


# ---------------------------------------------------------------------------
# 3. Extended graph generators (Scale-Free, GN, GNR, Directed ER)
# ---------------------------------------------------------------------------
original_create_graph = Sampler._create_graph

def patched_create_graph(self, n, weighted, directed, low=0.0, high=1.0, **kwargs):
    """Extended _create_graph supporting scale_free, gn, gnr, directed ER."""
    connected = kwargs.get('connected', True)

    if self._graph_generator == 'scale_free':
        n_val = self._select_parameter(n)
        alpha, beta, gamma = kwargs.get('alpha', 0.41), kwargs.get('beta', 0.54), kwargs.get('gamma', 0.05)
        if isinstance(alpha, list): alpha = alpha[0]
        if isinstance(beta, list): beta = beta[0]
        if isinstance(gamma, list): gamma = gamma[0]
        while True:
            # Generate scale free MultiDiGraph
            G = nx.scale_free_graph(n_val, alpha=alpha, beta=beta, gamma=gamma)
            # Convert to standard undirected Graph
            G = nx.Graph(G)
            # Remove self loops
            G.remove_edges_from(nx.selfloop_edges(G))
            if connected and not nx.is_connected(G):
                continue
            mat = nx.to_numpy_array(G)
            break
    elif self._graph_generator == 'gn':
        n_val = self._select_parameter(n)
        # GN graph is always a weakly connected tree inherently
        G = nx.gn_graph(n_val).to_undirected()
        mat = nx.to_numpy_array(G)
    elif self._graph_generator == 'gnr':
        n_val = self._select_parameter(n)
        p = kwargs.get('p', 0.5)
        if isinstance(p, list):
            p = self._select_parameter(p)
        # GNR graph is always a weakly connected tree inherently
        G = nx.gnr_graph(n_val, p=p).to_undirected()
        mat = nx.to_numpy_array(G)
    elif self._graph_generator is None or self._graph_generator == 'er':
        n_val = self._select_parameter(n)
        p_val = self._select_parameter(kwargs.get('p'), kwargs.get('p_range'))
        while True:
            G = nx.erdos_renyi_graph(n_val, p_val, directed=directed)
            if connected:
                if directed and not nx.is_weakly_connected(G):
                    continue
                elif not directed and not nx.is_connected(G):
                    continue
            mat = nx.to_numpy_array(G)
            break
    else:
        # We won't support directed WS or Delaunay properly yet
        mat = original_create_graph(self, n, weighted, False, low, high, **kwargs)

    n_mat = mat.shape[0]
    if weighted:
        weights = self._rng.uniform(low=low, high=high, size=(n_mat, n_mat))
        if not directed:
            weights *= np.transpose(weights)
            weights = np.sqrt(weights + 1e-3)
        mat = mat.astype(float) * weights
    return mat

Sampler._create_graph = patched_create_graph


# ---------------------------------------------------------------------------
# 4. BFS sampler: support directed graphs via global config
# ---------------------------------------------------------------------------
def patched_bfs_sample_data(self):
    generator_kwargs = self._get_graph_generator_kwargs()
    # Fetch global config via get_cfg_defaults to see if we should enforce directed graphs
    cfg = get_cfg_defaults()
    is_directed = cfg.DATA.get("DIRECTED", False)
    generator_kwargs.update({"directed": is_directed, "acyclic": False, "weighted": False})
    graph = self._create_graph(**generator_kwargs)
    source_node = self._rng.choice(graph.shape[0])
    return [graph, source_node]

BfsSampler._sample_data = patched_bfs_sample_data


# ---------------------------------------------------------------------------
# 5. Register bellman_ford in SAMPLERS (commented out in the library)
# ---------------------------------------------------------------------------
if 'bellman_ford' not in SAMPLERS:
    SAMPLERS['bellman_ford'] = BellmanFordSampler
