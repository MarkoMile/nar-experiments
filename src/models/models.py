# This file defines the SALSA-CLRS model architecture.

import torch.nn as nn
import torch
import torch_scatter
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import global_mean_pool, global_max_pool
from inspect import signature
from loguru import logger

from src.utils.utils import stack_hidden
from src.utils.utils import NaNException

#################
# PROCESSOR
#################

class PGN(pyg_nn.MessagePassing):
    """Adapted from https://github.com/google-deepmind/clrs/blob/64e016998f14305f94cf3f6d19ac9d7edc39a185/clrs/_src/processors.py#L330"""
    def __init__(self, in_channels, out_channels, aggr, mid_act=None, activation=nn.ReLU()):
        super(PGN, self).__init__(aggr=aggr)
        logger.info(f"PGN: in_channels: {in_channels}, out_channels: {out_channels}")
        self.in_channels = in_channels
        self.mid_channels = out_channels
        self.mid_act = mid_act
        self.out_channels = out_channels
        self.activation = activation

        # Message MLPs
        self.m_1 = nn.Linear(in_channels, self.mid_channels) # source node
        self.m_2 = nn.Linear(in_channels, self.mid_channels) # target node
        
        self.msg_mlp = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.mid_channels, self.mid_channels),
            nn.ReLU(),
            nn.Linear(self.mid_channels, self.mid_channels)
        )

        # Edge weight scaler
        self.edge_weight_scaler = nn.Linear(1, self.mid_channels)

        # Output MLP
        self.o1 = nn.Linear(in_channels, out_channels) # skip connection
        self.o2 = nn.Linear(self.mid_channels, out_channels)


    def forward(self, x, edge_index, edge_weight=None):
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        h_1 = self.o1(x)
        h_2 = self.o2(out)
        out = h_1 + h_2
        if self.activation is not None:
            out = self.activation(out)
        return out
    
    def message(self, x_j, x_i, edge_weight=None):
        # j is source, i is target
        msg_1 = self.m_1(x_j)
        msg_2 = self.m_2(x_i)
        
        msg = msg_1 + msg_2        
        if edge_weight is not None:
            msg_e = self.edge_weight_scaler(edge_weight.reshape(-1, 1))
            msg = msg + msg_e
        
        msg = self.msg_mlp(msg)

        if self.mid_act is not None:
            msg = self.mid_act(msg)

        return msg

def _get_processor(name):
    if name == "GCNConv":
        return pyg_nn.GCNConv
    elif name == "GINConv":
        return _gin_module
    elif name == "PGN":
        return PGN
    else:
        raise ValueError(f"Unknown processor {name}")

def _gin_module(in_channels, out_channels, eps=0, train_eps=False, layers=2, dropout=0.0, use_bn=False, aggr="add", **kwargs):
    mlp = nn.Sequential(
        nn.Linear(in_channels, out_channels),
    )
    if use_bn:
        logger.debug(f"Using batch norm in GIN module")
        mlp.add_module(f"bn_input", nn.BatchNorm1d(out_channels))
    for _ in range(layers-1):
        mlp.add_module(f"relu_{_}", nn.ReLU())
        mlp.add_module(f"linear_{_}", nn.Linear(out_channels, out_channels))
        if use_bn:
            logger.debug(f"Using batch norm in GIN module")
            mlp.add_module(f"bn_{_}", nn.BatchNorm1d(out_channels))
    if dropout > 0:
        mlp.add_module(f"dropout", nn.Dropout(dropout))
    return pyg_nn.GINConv(mlp, eps, train_eps, aggr=aggr)

class Processor(nn.Module):
    def __init__(self, cfg, has_randomness=False):
        super().__init__()
        self.cfg = cfg        
        processor_input = self.cfg.MODEL.HIDDEN_DIM*3 if self.cfg.MODEL.PROCESSOR_USE_LAST_HIDDEN else self.cfg.MODEL.HIDDEN_DIM*2
        if has_randomness:
            processor_input += 1
        # In "concat" mode, the encoded hint is a separate input channel rather than being
        # added into the hidden state — this prevents AR prediction errors from corrupting hidden.
        if getattr(self.cfg.MODEL, "HINT_INJECTION_MODE", "additive") == "concat":
            processor_input += self.cfg.MODEL.HIDDEN_DIM
            
        kwargs = self.cfg.MODEL.PROCESSOR.KWARGS[0].copy()
        self.core = _get_processor(self.cfg.MODEL.PROCESSOR.NAME)(in_channels=processor_input, out_channels=self.cfg.MODEL.HIDDEN_DIM, **kwargs)
        if self.cfg.MODEL.PROCESSOR.LAYERNORM.ENABLE:
            self.norm = pyg_nn.LayerNorm(self.cfg.MODEL.HIDDEN_DIM, mode=self.cfg.MODEL.PROCESSOR.LAYERNORM.MODE)
        
        self._core_requires_last_hidden = "last_hidden" in signature(self.core.forward).parameters

    def forward(self, input_hidden, hidden, last_hidden, batch_assignment, randomness=None, hint=None, **kwargs):
        stacked = stack_hidden(input_hidden, hidden, last_hidden, self.cfg.MODEL.PROCESSOR_USE_LAST_HIDDEN)
        if getattr(self.cfg.MODEL, "HINT_INJECTION_MODE", "additive") == "concat":
            # Hint is a dedicated input channel — wrong AR predictions don't touch hidden state.
            hint_input = hint if hint is not None else torch.zeros_like(input_hidden)
            stacked = torch.cat([stacked, hint_input], dim=-1)
        if randomness is not None:
            stacked = torch.cat((stacked, randomness.unsqueeze(1)), dim=-1)
        if self._core_requires_last_hidden:
            kwargs["last_hidden"] = last_hidden
        out = self.core(stacked, **kwargs)
        if self.cfg.MODEL.PROCESSOR.LAYERNORM.ENABLE:
            # norm
            out = self.norm(out, batch=batch_assignment)
        return out

    def has_edge_weight(self):
        return "edge_weight" in signature(self.core.forward).parameters
    
    def has_edge_attr(self):
        return "edge_attr" in signature(self.core.forward).parameters

#################
# ENCODER
#################

class NodePointerEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        # input_dim is ignored since pointers are just probabilities
        # We project the aggregated message into the hidden state
        self.lin = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, probabilities, hidden, edge_index):
        # probabilities: [E] (either soft probabilities or hard 1-hot)
        # hidden: [N, H] (current hidden state of nodes)
        # edge_index: [2, E]
        
        # 1. Get the hidden state of the target nodes (the nodes being pointed to)
        target_hidden = hidden[edge_index[1]] # [E, H]
        
        # 2. Weight by the pointer probability
        weighted_msg = target_hidden * probabilities.unsqueeze(-1) # [E, H]
        
        # 3. Aggregate back to the source nodes
        num_nodes = hidden.size(0)
        aggregated = torch_scatter.scatter_add(weighted_msg, edge_index[0], dim=0, dim_size=num_nodes) # [N, H]
        
        # 4. Project
        return self.lin(aggregated)

class NodeBaseEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.lin = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        x = self.lin(x)
        return x

_ENCODER_MAP = {
    ('node', 'scalar'): NodeBaseEncoder,
    ('node', 'mask'): NodeBaseEncoder,
    ('node', 'mask_one'): NodeBaseEncoder,
    ('node', 'pointer'): NodePointerEncoder,
}

class Encoder(nn.Module):
    def __init__(self, specs, hidden_dim=128):
        super().__init__()
        self.specs = specs
        self.hidden_dim = hidden_dim
        self.encoder = nn.ModuleDict()
        for k, v in specs.items():
            if k == "randomness": # randomness is not encoded
                continue
            stage, loc, type_, cat_dim = v
            if loc == 'edge':
                logger.debug(f'Ignoring edge encoder for {k}')
                continue
            elif stage == 'hint':
                logger.debug(f'Ignoring hint encoder for {k}')
                continue
            elif stage == 'output':
                logger.debug(f'Ignoring output encoder for {k}')
                continue
            else:
                # Input DIM currently hardcoded to 1
                self.encoder[k] = _ENCODER_MAP[(loc, type_)](1, hidden_dim)

    def forward(self, batch):
        hidden = None
        for key in batch.inputs:
            if key == "randomness":
                continue
            logger.debug(f"Encoding {key}")
            if key not in self.encoder:
                continue
            encoding = self.encoder[key](batch[key])
            # check of nan
            if torch.isnan(encoding).any():
                logger.warning(f"NaN in encoded hidden state")
                raise NaNException(f"NaN in encoded hidden state")
            if hidden is None:
                hidden = encoding
            else:
                hidden += encoding

        randomness = batch.randomness if "randomness" in batch.inputs else None
        return hidden, randomness

class HintEncoder(nn.Module):
    def __init__(self, specs, hidden_dim=128):
        super().__init__()
        self.specs = specs
        self.hidden_dim = hidden_dim
        self.encoder = nn.ModuleDict()
        for k, v in specs.items():
            stage, loc, type_, cat_dim = v
            if stage != 'hint':
                continue
            if loc == 'edge':
                logger.debug(f'Ignoring edge hint encoder for {k}')
                continue
            input_dim = 1
            if type_ == 'categorical':
                input_dim = cat_dim
                
            if (loc, type_) not in _ENCODER_MAP:
                logger.debug(f"Skipping {k} in HintEncoder: {(loc, type_)} not in _ENCODER_MAP")
                continue
                
            self.encoder[k] = _ENCODER_MAP[(loc, type_)](input_dim, hidden_dim)

    def forward(self, batch, step, current_hidden, pointer_only=None):
        """Encodes all ground-truth hints for a specific timestep.
        Only encodes node-level hints to sum with the node hidden state.
        
        Args:
            pointer_only: None = all hints (default), True = only pointer hints, False = only non-pointer hints
        """
        encoded_hidden = None
        for key in batch.hints:
            if key not in self.encoder:
                continue
            
            # Check if it's a node hint or edge hint (pointers are node loc but edge shape in hints)
            _, loc, type_, _ = self.specs[key]
            
            # Filter by pointer_only flag
            if pointer_only is True and type_ != 'pointer':
                continue
            if pointer_only is False and type_ == 'pointer':
                continue
            
            hint_step = batch[key][:, step]
            
            if type_ == 'pointer':
                # ground truth pointer is a one-hot vector across edges
                encoding = self.encoder[key](hint_step, current_hidden, batch.edge_index)
            else:
                encoding = self.encoder[key](hint_step)
                
            # check of nan
            if torch.isnan(encoding).any():
                logger.warning(f"NaN in encoded hint state for {key}")
                raise NaNException(f"NaN in encoded hint state for {key}")
                
            if encoded_hidden is None:
                encoded_hidden = encoding
            else:
                encoded_hidden += encoding
                
        return encoded_hidden
    
#################
# DECODER
#################

## Node decoders

class NodeBaseDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.lin = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, *args, **kwargs):
        x = self.lin(x)
        # Prevent any infinite or NaN values from escaping the decoder
        return torch.nan_to_num(x)

class NodeScalarDecoder(NodeBaseDecoder):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__(input_dim, hidden_dim)

    def forward(self, x, *args, **kwargs):
        out = super().forward(x).squeeze(-1)
        return out

class NodeMaskDecoder(NodeBaseDecoder):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__(input_dim, hidden_dim)

    def forward(self, x, *args, **kwargs):
        out = super().forward(x).squeeze(-1)
        return out

class NodeMaskOneDecoder(NodeBaseDecoder):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__(input_dim, hidden_dim)

    def forward(self, x, batch_assignment, **kwargs):
        out = super().forward(x) # N x 1

        out = torch_scatter.scatter_log_softmax(out, batch_assignment, dim=0)
        return torch.nan_to_num(out)


class NodeCategoricalDecoder(NodeBaseDecoder):
    def __init__(self, input_dim, hidden_dim=128, use_fp64=True):
        super().__init__(input_dim, hidden_dim)
        self.use_fp64 = use_fp64

    def forward(self, x, batch_assignment, **kwargs):
        out = super().forward(x) # N x C
        if self.use_fp64:
            out = out.float()
        out = torch.log_softmax(out, dim=-1)
        return torch.nan_to_num(out)



#### Edge decoders

class BaseEdgeDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, use_fp64=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.use_fp64 = use_fp64
        self.source_lin = nn.Linear(hidden_dim, hidden_dim)
        self.target_lin = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, hiddens, edge_index):
        zs = self.source_lin(hiddens) # N x H
        zt = self.target_lin(hiddens) # N x H
        # Optional fp64 path improves numerical stability but can be very memory-heavy on large batches.
        if self.use_fp64:
            zs = zs.double()
            zt = zt.double()
        out = (zs[edge_index[0]] * zt[edge_index[1]]).sum(dim=-1) / (self.hidden_dim ** 0.5)
        # out = torch.clamp(out, min=-50, max=50)
        return out
    
class EdgeMaskDecoder(BaseEdgeDecoder):
    def __init__(self, input_dim, hidden_dim=128, use_fp64=True):
        super().__init__(input_dim, hidden_dim, use_fp64=use_fp64)

    def forward(self, hiddens, edge_index, **kwargs):
        out = super().forward(hiddens, edge_index)
        if self.use_fp64:
            out = out.float()
        out = out.squeeze(-1)
        # out = torch.clamp(out, min=-50, max=50)
        return torch.nan_to_num(out)
    
class NodePointerDecoder(BaseEdgeDecoder):
    def __init__(self, input_dim, hidden_dim=128, use_fp64=True):
        super().__init__(input_dim, hidden_dim, use_fp64=use_fp64)

    def forward(self, hiddens, edge_index, **kwargs):
        z =  super().forward(hiddens, edge_index) # E
        # per node outgoing softmax
        z = torch_scatter.scatter_log_softmax(z, edge_index[0], dim=0)
        if self.use_fp64:
            z = z.float()
        return torch.nan_to_num(z)

#### Graph decoders

class GraphBaseDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.lin = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, batch_assignment, **kwargs):
        x = self.lin(x)
        out = global_mean_pool(x, batch_assignment)
        # out = torch.clamp(out, min=-50, max=50)
        return torch.nan_to_num(out.squeeze(-1))
    
class GraphMaskDecoder(GraphBaseDecoder):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__(input_dim, hidden_dim)

    def forward(self, x, batch_assignment, **kwargs):
        out = super().forward(x, batch_assignment)
        # out = torch.clamp(out, min=-50, max=50)
        return torch.nan_to_num(out)

class GraphCategoricalDecoder(GraphBaseDecoder):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__(input_dim, hidden_dim)

    def forward(self, x, batch_assignment, **kwargs):
        out = super().forward(x, batch_assignment)
        out = torch.log_softmax(out, dim=-1)
        return torch.nan_to_num(out)
    

_DECODER_MAP = {
    ('node', 'scalar'): NodeScalarDecoder,
    ('node', 'mask'): NodeMaskDecoder,
    ('node', 'mask_one'): NodeMaskOneDecoder,
    ('node', 'pointer'): NodePointerDecoder,
    ('node', 'categorical'): NodeCategoricalDecoder,
    ('edge', 'mask'): EdgeMaskDecoder,
    ('edge', 'scalar'): BaseEdgeDecoder,
    ('graph', 'scalar'): GraphBaseDecoder,  
    ('graph', 'mask'): GraphMaskDecoder,
    ('graph', 'categorical'): GraphCategoricalDecoder,
}
    
class Decoder(nn.Module):
    def __init__(self, specs, hidden_dim=128, no_hint=False, edge_decoder_fp64=True):
        super().__init__()
        self.specs = specs
        self.hidden_dim = hidden_dim
        self.decoder = nn.ModuleDict()
        for k, v in specs.items():
            stage, loc, type_, cat_dim = v
            if no_hint and stage == 'hint':
                logger.debug(f'Ignoring hint decoder for {k}')
                continue
            if stage == 'input':
                logger.debug(f'Ignoring input decoder for {k}')
                continue
            if stage == 'hint':
                k = k.removesuffix('_h')
            
            input_dim = 1
            if type_ == 'categorical':
                input_dim = cat_dim

            if k not in self.decoder:
                decoder_cls = _DECODER_MAP[(loc, type_)]
                if issubclass(decoder_cls, BaseEdgeDecoder) or decoder_cls == NodeCategoricalDecoder:
                    self.decoder[k] = decoder_cls(input_dim, hidden_dim, use_fp64=edge_decoder_fp64)
                else:
                    self.decoder[k] = decoder_cls(input_dim, hidden_dim)

    def forward(self, hidden, batch, stage):
        output = {}
        for key in getattr(batch, stage):
            if stage == 'hints':
                dkey = key.removesuffix('_h')
            else:
                dkey = key

            output[key] = self.decoder[dkey](hidden, edge_index=batch.edge_index, batch_assignment=batch.batch)
        return output

    
def grab_outputs(hints, batch):
    """This function grabs the outputs from the batch and returns them in the same format as the hints"""
    output = {}
    for k in hints:
        k_out = k.replace('_h', '')
        if k_out in batch.outputs:
            output[k_out] = hints[k]
    return output

def output_mask(batch, step):
    final_node_idx = (batch.length[batch.batch]-1)

    masks = {}
    for key in batch.outputs:
        if key in batch.edge_attrs():
            final_edge_idx = final_node_idx[batch.edge_index[0]]
            masks[key] = final_edge_idx == step
        elif key in batch.node_attrs():
            masks[key] = final_node_idx == step
        else:
            # graph attribute
            masks[key] = batch.length == step + 1
    return masks

#################
# Encode-Process-Decode
#
# [ALL 3 PUT TOGETHER]
#################

def stack_hints(hints):
    return {k: torch.stack([hint[k] for hint in hints], dim=-1) for k in hints[0]} if hints else {}

class EncodeProcessDecode(torch.nn.Module):
    def __init__(self, specs, cfg):
        super().__init__()
        self.cfg = cfg
        self.specs = specs
        self.has_randomness = 'randomness' in specs
        self.processor = Processor(cfg, self.has_randomness)
        self.encoder = Encoder(specs, self.cfg.MODEL.HIDDEN_DIM)
        if self.cfg.MODEL.TEACHER_FORCING.ENABLE or self.cfg.MODEL.AUTOREGRESSIVE.ENABLE:
            self.hint_encoder = HintEncoder(specs, self.cfg.MODEL.HIDDEN_DIM)
        self.residual_norm = torch.nn.LayerNorm(self.cfg.MODEL.HIDDEN_DIM)

        if self.cfg.MODEL.GRU.ENABLE:
            self.gru = torch.nn.GRUCell(self.cfg.MODEL.HIDDEN_DIM, self.cfg.MODEL.HIDDEN_DIM)

        # self.processor = torch.compile(self.processor, dynamic=True)
        # if self.cfg.MODEL.GRU.ENABLE:
        #     self.gru = torch.compile(self.gru, dynamic=True)
        decoder_input = self.cfg.MODEL.HIDDEN_DIM*3 if self.cfg.MODEL.DECODER_USE_LAST_HIDDEN else self.cfg.MODEL.HIDDEN_DIM*2
        self.decoder = Decoder(
            specs,
            decoder_input,
            no_hint=self.cfg.TRAIN.LOSS.HINT_LOSS_WEIGHT == 0.0,
            edge_decoder_fp64=getattr(self.cfg.MODEL, "EDGE_DECODER_FP64", False),
        )
        logger.debug(f"Decoder: {self.cfg.TRAIN.LOSS.HINT_LOSS_WEIGHT == 0.0}")

        self.training_progress = 0.0  # Set by Lightning module; 0 to 1 over training

        if not self.processor.has_edge_weight() and not self.processor.has_edge_attr():
            if "A" in specs:
                logger.warning(f"Processor {self.cfg.MODEL.PROCESSOR.NAME} does neither support edge_weight nor edge_attr, but the algorithm requires edge weights.")
                raise ValueError(f"Processor {self.cfg.MODEL.PROCESSOR.NAME} does neither support edge_weight nor edge_attr, but the algorithm requires edge weights.")
        elif self.processor.has_edge_weight():
            self.edge_weight_name = "edge_weight"
        elif self.processor.has_edge_attr():
            self.edge_weight_name = "edge_attr"
        
    def _get_hint_dropout(self):
        """Return the current hint dropout probability, accounting for curriculum."""
        cur = self.cfg.MODEL.TEACHER_FORCING.CURRICULUM
        if cur.ENABLE:
            return cur.START_DROPOUT + (cur.END_DROPOUT - cur.START_DROPOUT) * self.training_progress
        return self.cfg.MODEL.TEACHER_FORCING.HINT_DROPOUT

    def process_weights(self, batch):
        if self.edge_weight_name == "edge_attr":
            return batch.weights.unsqueeze(-1).type(torch.float32)
        else:
            return batch.weights
        
    def forward(self, batch):
        input_hidden, randomness = self.encoder(batch)
        max_len = batch.length.max().item()
        hints = []
        output = None

        # Process for length
        hidden = input_hidden
        noise_std = self.cfg.MODEL.LATENT_NOISE_STD
        last_hidden_mode = getattr(self.cfg.MODEL, "LAST_HIDDEN_MODE", "current")  # "current" or "previous"
        last_hidden_carry = None  # used only in "previous" mode
        
        use_teacher_forcing = self.cfg.MODEL.TEACHER_FORCING.ENABLE and self.training
        use_autoregressive = self.cfg.MODEL.AUTOREGRESSIVE.ENABLE
        # Fair ablation switch: when POINTER=False, exclude pointer hints from both
        # AR and TF hint injection paths regardless of whether AR is enabled.
        disable_pointer_hints = not getattr(self.cfg.MODEL.AUTOREGRESSIVE, "POINTER", False)
        
        for step in range(max_len):
            # 1. Inject hints (Teacher Forcing or Autoregressive)
            encoded_hint_gt = None
            encoded_hint_ar = None

            if use_teacher_forcing:
                if step == 0:
                    encoded_hint_gt = torch.zeros_like(hidden)
                else:
                    # Fair comparison mode: if pointer AR is disabled, remove pointer hints from TF too.
                    tf_pointer_only = False if disable_pointer_hints else None
                    encoded_hint_gt = self.hint_encoder(batch, step - 1, hidden, pointer_only=tf_pointer_only)
                    if encoded_hint_gt is None:
                        encoded_hint_gt = torch.zeros_like(hidden)
            
            if use_autoregressive:
                if step == 0:
                    encoded_hint_ar = torch.zeros_like(hidden)
                else:
                    prev_hints = hints[-1]
                    encoded_hint_ar = None
                    for key in self.hint_encoder.encoder.keys():
                        if key not in prev_hints:
                            continue
                        
                        # Detach AR predictions to avoid BPTT through the unrolled loop
                        _, loc, type_, _ = self.specs[key]
                        
                        # Skip pointer hints when pointer hints are disabled for this run.
                        if disable_pointer_hints and type_ == 'pointer':
                            continue
                        
                        raw_pred = prev_hints[key].detach()
                        
                        if type_ == 'pointer':
                            if getattr(self.cfg.MODEL.AUTOREGRESSIVE, "POINTER_MODE", "soft") == 'hard':
                                # Hard argmax: pick best neighbor per source node → one-hot
                                # raw_pred is [E] log-softmax grouped by edge_index[0]
                                hard_pred = torch.zeros_like(raw_pred)
                                # For each source node, find the edge with max log-prob
                                num_nodes = hidden.size(0)
                                max_vals = torch_scatter.scatter_max(raw_pred, batch.edge_index[0], dim=0, dim_size=num_nodes)[0]
                                hard_pred = (raw_pred == max_vals[batch.edge_index[0]]).float()
                                key_encoding = self.hint_encoder.encoder[key](hard_pred, hidden, batch.edge_index)
                            else:
                                # Soft: convert log-softmax → probabilities
                                soft_pred = torch.exp(raw_pred)
                                key_encoding = self.hint_encoder.encoder[key](soft_pred, hidden, batch.edge_index)
                        else:
                            if type_ == 'categorical' or type_ == 'mask_one':
                                soft_pred = torch.exp(raw_pred)
                            elif type_ == 'mask':
                                soft_pred = torch.sigmoid(raw_pred)
                            else: # scalar
                                soft_pred = raw_pred
                            key_encoding = self.hint_encoder.encoder[key](soft_pred)
                            
                        if encoded_hint_ar is None:
                            encoded_hint_ar = key_encoding
                        else:
                            encoded_hint_ar += key_encoding
                            
                    if encoded_hint_ar is None:
                        encoded_hint_ar = torch.zeros_like(hidden)

            encoded_hint = None
            if use_teacher_forcing and use_autoregressive:
                # True Scheduled Sampling: Choose Autoregressive over GT with probability `dropout_prob`
                dropout_prob = self._get_hint_dropout()
                if dropout_prob > 0.0 and self.training:
                    # Graph-level mask: sample once per graph, apply to all nodes in that graph
                    num_graphs = batch.batch.max().item() + 1
                    graph_mask = (torch.rand(num_graphs, 1, device=hidden.device) > dropout_prob).float()
                    mask = graph_mask[batch.batch]  # Broadcast back to node level
                    encoded_hint = encoded_hint_gt * mask + encoded_hint_ar * (1.0 - mask)
                else:
                    encoded_hint = encoded_hint_gt
            elif use_teacher_forcing:
                # Only Teacher Forcing: Zero out hints with probability `dropout_prob`
                dropout_prob = self._get_hint_dropout()
                if dropout_prob > 0.0 and self.training:
                    # Graph-level mask: sample once per graph, apply to all nodes in that graph
                    num_graphs = batch.batch.max().item() + 1
                    graph_mask = (torch.rand(num_graphs, 1, device=encoded_hint_gt.device) > dropout_prob).float()
                    mask = graph_mask[batch.batch]  # Broadcast back to node level
                    # Do NOT scale when zeroing out either, to match evaluation distribution
                    encoded_hint = encoded_hint_gt * mask
                else:
                    encoded_hint = encoded_hint_gt
            elif use_autoregressive:
                # Only Autoregressive (pointer hints excluded if POINTER=False → zero pointer signal)
                encoded_hint = encoded_hint_ar

            hint_injection_mode = getattr(self.cfg.MODEL, "HINT_INJECTION_MODE", "additive")
            if hint_injection_mode == "concat":
                # Hint is passed as a dedicated processor input channel — hidden stays clean.
                processor_hint = encoded_hint  # may be None; Processor.forward handles that with zeros
            else:
                # "additive" (default): inject hint directly into hidden before processor
                if encoded_hint is not None:
                    hidden = hidden + encoded_hint
                processor_hint = None

            # 2. Inject latent Gaussian noise during training to prevent hidden state drift
            if self.training and noise_std > 0:
                hidden = hidden + torch.randn_like(hidden) * noise_std
            
            if last_hidden_mode == "previous":
                # Carry previous step's GRU output — processor sees 3 different inputs
                last_hidden = last_hidden_carry if last_hidden_carry is not None else hidden
            else:
                # SALSA-CLRS default: snapshot current hidden before processor
                last_hidden = hidden
            msg_passing_steps = getattr(self.cfg.MODEL, "MSG_PASSING_STEPS", 1)
            for _ in range(msg_passing_steps):
                processed = self.processor(input_hidden, hidden, last_hidden, hint=processor_hint, randomness=randomness[:, step] if randomness is not None else None, edge_index=batch.edge_index, batch_assignment=batch.batch, **{self.edge_weight_name: self.process_weights(batch) for _ in range(1) if hasattr(batch, 'weights') })
                if self.cfg.MODEL.PROCESSOR.RESIDUAL:
                    hidden = self.residual_norm(hidden + processed)
                else:
                    hidden = processed
                if self.cfg.MODEL.GRU.ENABLE:
                    hidden = self.gru(hidden, last_hidden)
            if last_hidden_mode == "previous":
                last_hidden_carry = hidden  # save GRU output for next step
            if self.cfg.TRAIN.LOSS.HINT_LOSS_WEIGHT > 0.0:
                hints.append(self.decoder(stack_hidden(input_hidden, hidden, last_hidden, self.cfg.MODEL.DECODER_USE_LAST_HIDDEN), batch, 'hints'))

            # Check if output needs to be constructed
            if (batch.length == step+1).sum() > 0:
                # Decode outputs
                if self.cfg.TRAIN.LOSS.HINT_LOSS_WEIGHT > 0.0:
                    # The last hint is the output, no need to decode again, its the same decoder
                    output_step = grab_outputs(hints[-1], batch)
                else:
                    output_step = self.decoder(stack_hidden(input_hidden, hidden, last_hidden, self.cfg.MODEL.DECODER_USE_LAST_HIDDEN), batch, 'outputs')
                
                # Mask output
                mask = output_mask(batch, step)   
                if output is None:
                    output = {k: torch.zeros_like(output_step[k]) for k in output_step}
                    for k in output_step:
                        output[k][mask[k]] = output_step[k][mask[k]]
                else:
                    for k in output_step:
                        output[k][mask[k]] = output_step[k][mask[k]]

        hints = stack_hints(hints)

        return output, hints, hidden