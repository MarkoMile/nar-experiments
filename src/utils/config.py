from yacs.config import CfgNode as CN


_C = CN()
_C.ALGORITHM = "bfs"
_C.RUN_NAME = "test"
# -----------------------------------------------------------------------------
# Model

_C.MODEL = CN()
_C.MODEL.HIDDEN_DIM = 128
_C.MODEL.MSG_PASSING_STEPS = 1

_C.MODEL.PROCESSOR = CN()
_C.MODEL.PROCESSOR.RESIDUAL = False
_C.MODEL.PROCESSOR.LAYERNORM = CN()
_C.MODEL.PROCESSOR.LAYERNORM.ENABLE = False
_C.MODEL.PROCESSOR.LAYERNORM.MODE = "graph"
_C.MODEL.PROCESSOR.NAME = "GINConv"

_C.MODEL.PROCESSOR.KWARGS = [{}] # dict not allowed so we use list of dict and just first element is used

_C.MODEL.DECODER_USE_LAST_HIDDEN = False
_C.MODEL.PROCESSOR_USE_LAST_HIDDEN = False
_C.MODEL.LAST_HIDDEN_MODE = "current"  # "current" = set before processor (SALSA-CLRS default), "previous" = carry from previous step's GRU output

_C.MODEL.LATENT_NOISE_STD = 0.0  # Gaussian noise σ injected into hidden state (0.0 = disabled)

_C.MODEL.GRU = CN()
_C.MODEL.GRU.ENABLE = False

_C.MODEL.TEACHER_FORCING = CN()
_C.MODEL.TEACHER_FORCING.ENABLE = False
_C.MODEL.TEACHER_FORCING.HINT_DROPOUT = 0.0
_C.MODEL.TEACHER_FORCING.CURRICULUM = CN()
_C.MODEL.TEACHER_FORCING.CURRICULUM.ENABLE = False
_C.MODEL.TEACHER_FORCING.CURRICULUM.START_DROPOUT = 0.0
_C.MODEL.TEACHER_FORCING.CURRICULUM.END_DROPOUT = 0.5

_C.MODEL.AUTOREGRESSIVE = CN()
_C.MODEL.AUTOREGRESSIVE.ENABLE = False
_C.MODEL.AUTOREGRESSIVE.POINTER = True  # With AR enabled: True = include pi_h in TF+AR paths, False = exclude pi_h from both TF+AR paths.
_C.MODEL.AUTOREGRESSIVE.POINTER_MODE = "soft"  # "soft" = exp(log_softmax) probs, "hard" = argmax one-hot per source node

# How encoded hints are injected into the processor:
#   "additive" (default): hidden += encoded_hint before processor (corrupts hidden state)
#   "concat": hint is concatenated as a separate input channel to the processor (CLRS-30 style)
_C.MODEL.HINT_INJECTION_MODE = "additive"

# -----------------------------------------------------------------------------
# Training

_C.TRAIN = CN()
_C.TRAIN.PRECISION = "16-mixed"
_C.TRAIN.ENABLE = True
_C.TRAIN.LOAD_CHECKPOINT = None

_C.TRAIN.BATCH_SIZE = 512
_C.TRAIN.NUM_WORKERS = 8
_C.TRAIN.MAX_EPOCHS = 200
_C.TRAIN.GRADIENT_CLIP_VAL = 1.0
_C.TRAIN.EARLY_STOPPING_PATIENCE = 200
_C.TRAIN.CHECKPOINT_MONITOR = "val/graph_accuracy/{val_nickname}"
_C.TRAIN.CHECKPOINT_MONITOR_MODE = "max"

_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = "adamw"
_C.TRAIN.OPTIMIZER.LR = 1e-3
_C.TRAIN.OPTIMIZER.WEIGHT_DECAY = 1e-2

_C.TRAIN.SCHEDULER = CN()
_C.TRAIN.SCHEDULER.NAME = "ReduceLROnPlateau"
_C.TRAIN.SCHEDULER.ENABLE = False
_C.TRAIN.SCHEDULER.MONITOR = "train/loss"
_C.TRAIN.SCHEDULER.PARAMS = [{"mode": "min", "factor": 0.1, "patience": 10}]
_C.TRAIN.SCHEDULER.INTERVAL = "epoch"  # "epoch" or "step"

# Normalize-and-Project (NaP): periodically project weights to initial Frobenius norm
_C.TRAIN.WEIGHT_PROJECTION = CN()
_C.TRAIN.WEIGHT_PROJECTION.ENABLE = False
_C.TRAIN.WEIGHT_PROJECTION.EVERY_N_STEPS = 1  # project every N optimizer steps
_C.TRAIN.WEIGHT_PROJECTION.SCALE_DECAY = False  # apply separate weight decay to LayerNorm scale params
_C.TRAIN.WEIGHT_PROJECTION.SCALE_DECAY_FACTOR = 1.0  # multiplier on WD for LayerNorm scales (1.0 = same as main WD)

_C.TRAIN.LOSS = CN()
_C.TRAIN.LOSS.OUTPUT_LOSS_WEIGHT = 1.0
_C.TRAIN.LOSS.HINT_LOSS_WEIGHT = 1.0
_C.TRAIN.LOSS.HIDDEN_LOSS_WEIGHT = 0.0
_C.TRAIN.LOSS.HIDDEN_LOSS_TYPE = "l2"


# -----------------------------------------------------------------------------
# Testing

_C.TEST = CN()
_C.TEST.BATCH_SIZE = 8


# -----------------------------------------------------------------------------
# Data

_C.DATA = CN()

_C.DATA.ROOT = "./data"
_C.DATA.DIRECTED = False

_C.DATA.TRAIN = CN()
_C.DATA.TRAIN.NUM_SAMPLES = [10000]
_C.DATA.TRAIN.GRAPH_GENERATOR = ["er"]
_C.DATA.TRAIN.GENERATOR_PARAMS = [{"p": [0.5], "n": 16}]
_C.DATA.TRAIN.START_EPOCH = [0]

_C.DATA.VAL = CN()
_C.DATA.VAL.NUM_SAMPLES = 1000
_C.DATA.VAL.GRAPH_GENERATOR = ["er"]
_C.DATA.VAL.GENERATOR_PARAMS = [{"p": [0.5], "n": 16}]
_C.DATA.VAL.NICKNAME = ["er_mid"]

_C.DATA.TEST = CN()
_C.DATA.TEST.NUM_SAMPLES = 1000
_C.DATA.TEST.GRAPH_GENERATOR = ["er", "er"]
_C.DATA.TEST.GENERATOR_PARAMS = [{"p": [0.5, 0.6, 0.7, 0.8, 0.9], "n": 16}, {"p": [0.1, 0.2, 0.3, 0.4, 0.5], "n":16}]
_C.DATA.TEST.NICKNAME = ["er_mid", "er_hard"]


# -----------------------------------------------------------------------------
# Logging

_C.LOGGING = CN()
_C.LOGGING.WANDB = CN()
_C.LOGGING.WANDB.PROJECT = "nar-experiments"
_C.LOGGING.WANDB.GROUP = "default"

# -----------------------------------------------------------------------------

def get_cfg_defaults():
    return _C.clone()

def load_cfg(cfg_path):
    cfg = get_cfg_defaults()
    cfg.merge_from_file(cfg_path)
    return cfg