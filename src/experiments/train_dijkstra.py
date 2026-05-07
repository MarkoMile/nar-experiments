# Training entry point for the Dijkstra task.
#
# This is a thin wrapper around the algorithm-agnostic training pipeline.
# The algorithm is determined by the config file's ALGORITHM field.
#
# Usage:
#   python -m src.experiments.train_dijkstra --cfg src/configs/dijkstra/bestmodel.yml --enable-wandb

import os
import sys

# Prevent TensorFlow (imported transitively by dm-clrs) from pre-allocating all GPU memory
os.environ["CUDA_VISIBLE_DEVICES_TF"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# Reduce CUDA fragmentation for variable-size graph batches
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from loguru import logger

# Add project root to sys.path so absolute imports work when running directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

os.environ["OMP_NUM_THREADS"] = "4"
torch.set_num_threads(4)

logger.remove()
logger.add(sys.stderr, level="INFO")

from src.experiments.train import setup_and_train

if __name__ == '__main__':
    setup_and_train()
