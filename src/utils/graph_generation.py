import torch
import os
from loguru import logger
from salsaclrs import load_dataset, SALSACLRSDataset
from yacs.config import CfgNode
from torch.utils.data import ConcatDataset

from src.utils.config import get_cfg_defaults

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
