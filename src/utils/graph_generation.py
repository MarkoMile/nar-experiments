import torch
import os
from salsaclrs import load_dataset
from yacs.config import CfgNode

from src.utils.config import get_cfg_defaults

# Monkeypatch torch.load to force weights_only=False globally.
# This is necessary because salsaclrs uses older pickle method and we don't control the library code
# to pass weights_only=False down.
# This must be done BEFORE importing salsaclrs (or any module that imports it) 
# as it might bind torch.load early.
original_torch_load = torch.load

def unsafe_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
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
        dataset = load_dataset(algorithm=algorithm, split=split, local_dir=root)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise e

    return dataset
