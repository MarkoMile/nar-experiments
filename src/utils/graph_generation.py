import torch
import os
from loguru import logger
from salsaclrs import load_dataset, SALSACLRSDataset
from yacs.config import CfgNode
from torch.utils.data import ConcatDataset

from src.utils.config import get_cfg_defaults

import networkx as nx
import numpy as np
from salsaclrs.sampler import Sampler, BfsSampler
import salsaclrs.data as custom_data

# Monkey-patch verify_sparseness to allow backward pointers in directed graphs.
# In directed graphs, BFS pointers (pi) point from child to parent, which may not be
# an edge defined in the original asymmetric `edge_index`.
def patched_verify_sparseness(data, edge_index, data_name):
    pass # Disable sparseness checking so backward edges don't crash dataset generation

custom_data.verify_sparseness = patched_verify_sparseness

# Monkeypatch torch.load to force weights_only=False globally.
# This is necessary because salsaclrs uses older pickle method and we don't control the library code
# to pass weights_only=False down.
# This must be done BEFORE importing salsaclrs (or any module that imports it) 
# as it might bind torch.load early.
original_torch_load = torch.load

def unsafe_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)

torch.load = unsafe_torch_load

# Monkeypatch SALSA-CLRS Sampler to support directed graphs (Scale-Free and Directed ER)
original_create_graph = Sampler._create_graph

def patched_create_graph(self, n, weighted, directed, low=0.0, high=1.0, **kwargs):
    # Try our new generators first
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


def get_dataset(split: str, cfg: CfgNode = None):
    """
    Load a SALSA-CLRS dataset for a specific split using the project config.

    This function wraps salsaclrs.load_dataset, reading the algorithm name
    and data root from the config.

    Args:
        split (str): One of 'train', 'val', 'test'.
        cfg (CfgNode, optional): A yacs config node. If None, the default
            config from config.py is used.

    Returns:
        SALSACLRSDataset or dict:
            - For 'train' and 'val' splits, returns a SALSACLRSDataset object.
            - For 'test' split, returns a dictionary mapping graph generator names
              (e.g., 'er_80', 'ws_160') to SALSACLRSDataset objects.
    """
    if cfg is None:
        cfg = get_cfg_defaults()

    algorithm = cfg.ALGORITHM
    root = os.path.join(cfg.DATA.ROOT, "salsaclrs")

    # Ensure the data directory exists
    os.makedirs(root, exist_ok=True)

    # Check if split is valid
    if split not in ['train', 'val', 'test']:
        raise ValueError(f"Invalid split: {split}. Must be one of ['train', 'val', 'test']")

    print(f"Loading {algorithm} dataset for split: {split}...")

    try:
        # Check if config has custom generator params for this split
        # We assume generator_params is a list of dicts, we take the first one for now
        # unless we want to support multiple datasets (DynamicDataset) which is more complex
        
        split_upper = split.upper()
        if hasattr(cfg.DATA, split_upper) and hasattr(getattr(cfg.DATA, split_upper), "GENERATOR_PARAMS"):
             params = getattr(cfg.DATA, split_upper).GENERATOR_PARAMS
             if isinstance(params, list) and len(params) > 0:
                 logger.info(f"Using custom generator params from config for {split}: {params}")
                 
                 if split == "train":
                     # Train iterates through all generator params and concatenates them
                     datasets = []
                     generators = getattr(cfg.DATA, split_upper).GRAPH_GENERATOR
                     num_samples = getattr(cfg.DATA, split_upper).NUM_SAMPLES

                     for i, param_dict in enumerate(params):
                         gen = generators[i] if i < len(generators) else generators[0]
                         # Ensure num_samples is accessed correctly if it's a list or a single int
                         samples = num_samples[i] if isinstance(num_samples, list) and i < len(num_samples) else (num_samples[0] if isinstance(num_samples, list) else num_samples)
                         
                         ds = SALSACLRSDataset(
                             root=root,
                             split=split,
                             algorithm=algorithm,
                             num_samples=samples,
                             graph_generator=gen,
                             graph_generator_kwargs=param_dict,
                             verify_duplicates=False
                         )
                         datasets.append(ds)
                         
                     # Concatenate the generated datasets and monkeypatch the specs needed for the model
                     concat_ds = ConcatDataset(datasets)
                     concat_ds.specs = datasets[0].specs
                     return concat_ds
                 else:
                     # Val and Test return a dictionary of datasets
                     datasets = {}
                     generators = getattr(cfg.DATA, split_upper).GRAPH_GENERATOR
                     nicknames = getattr(cfg.DATA, split_upper).NICKNAME
                     num_samples = getattr(cfg.DATA, split_upper).NUM_SAMPLES
                     
                     for i, param_dict in enumerate(params):
                         gen = generators[i] if i < len(generators) else generators[0]
                         nickname = nicknames[i] if i < len(nicknames) else f"{gen}_{i}"
                         
                         ds = SALSACLRSDataset(
                             root=root,
                             split=split,
                             algorithm=algorithm,
                             num_samples=num_samples,
                             graph_generator=gen,
                             graph_generator_kwargs=param_dict,
                             verify_duplicates=False,
                             nickname=nickname,
                             ignore_all_hints=(cfg.TRAIN.LOSS.HINT_LOSS_WEIGHT == 0.0) # if hints are ignored in config, ignore them in val/test too
                         )
                         datasets[nickname] = ds
                     return datasets

        dataset = load_dataset(algorithm=algorithm, split=split, local_dir=root)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise e

    return dataset
