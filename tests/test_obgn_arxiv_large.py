"""
Script to evaluate a trained model checkpoint on the ogbn-arxiv dataset subgraphs.

Usage:
    python tests/test_obgn_arxiv_large.py --ckpt path/to/model.ckpt
"""

import os
import sys
import argparse
import torch
import numpy as np
import lightning.pytorch as pl

# Add project root to sys.path so absolute imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Module-level flag: when True, use actual BFS depth instead of worst-case N.
# Trades methodological correctness for speed on very large graphs.
USE_FAST_ITERATIONS = False

# ==============================================================================
# MONKEYPATCH: salsaclrs.data.to_sparse_data memory leak fix
# Bypasses the O(N^2) memory allocation when converting Node Pointers to
# one-hot matrices. Crucial for massive graphs (like Arxiv 169k nodes).
# ==============================================================================
def efficient_to_sparse_data(inputs, hints, outputs, use_hints=True):
    import clrs
    from scipy.sparse import coo_matrix, issparse
    from scipy.sparse import eye as sp_eye
    from torch_geometric.utils.convert import from_scipy_sparse_matrix
    from salsaclrs.data import infer_type, verify_sparseness, pointer_to_one_hot, to_torch, CLRSData
    
    data_dict = {}
    input_attributes = []
    hint_attributes = []
    output_attributes = []
    if use_hints and len(hints) > 0:
        data_dict['length'] = hints[0].data.shape[0]
    elif USE_FAST_ITERATIONS:
        # Fast mode: compute actual BFS depth from the pi (predecessor) array.
        # Faster but not methodologically correct (requires knowing the answer).
        pi_arr = outputs[0].data[0].astype(int)
        n = len(pi_arr)
        depths = np.zeros(n, dtype=int)
        for node in range(n):
            d, cur = 0, node
            while pi_arr[cur] != cur:
                d += 1
                cur = pi_arr[cur]
                if d > n:
                    break
            depths[node] = d
        data_dict['length'] = int(depths.max()) + 1
    else:
        # Default: worst-case iterations (N), matching DeepMind NAR methodology.
        n = outputs[0].data[0].shape[0]
        data_dict['length'] = n
    # first get the edge index
    for dp in inputs:
        if dp.name == "adj":
            edge_index, _ = from_scipy_sparse_matrix(coo_matrix(dp.data[0]))
            data_dict['edge_index'] = edge_index
            
    # Parse inputs
    for dp in inputs:
        if dp.name == "adj":
            continue
        elif dp.name == "A":
            A_data = dp.data[0]
            if issparse(A_data):
                unique_values = np.unique(A_data.data)  # only nonzero values
                # For sparse, 0 is implicit, so check if all nonzeros are 1
                is_weighted = not (unique_values.size == 1 and unique_values[0] == 1.0)
            else:
                unique_values = np.unique(A_data)
                is_weighted = unique_values.size != 2 or not np.all(unique_values == np.array([0,1]))
            if is_weighted:
                if issparse(A_data):
                    A_with_self = A_data + sp_eye(A_data.shape[0], dtype=float)
                    edge_weights = np.asarray(A_with_self[data_dict["edge_index"][0], data_dict["edge_index"][1]]).flatten()
                else:
                    edge_weights = (A_data + np.eye(A_data.shape[0]))[data_dict["edge_index"][0], data_dict["edge_index"][1]]
                data_dict["weights"] = infer_type("A", edge_weights)
        elif dp.location == clrs.Location.EDGE:
            verify_sparseness(dp.data[0], data_dict["edge_index"], dp.name)
            sliced_edge_data = dp.data[0][data_dict["edge_index"][0], data_dict["edge_index"][1]]
            if hasattr(sliced_edge_data, "toarray"):
                sliced_edge_data = sliced_edge_data.toarray()
            if isinstance(sliced_edge_data, np.matrix):
                sliced_edge_data = np.asarray(sliced_edge_data).flatten()
            elif isinstance(sliced_edge_data, np.ndarray):
                sliced_edge_data = sliced_edge_data.flatten()
            data_dict[dp.name] = infer_type(dp.type_, sliced_edge_data)
            input_attributes.append(dp.name)
        elif dp.location == clrs.Location.NODE:
            if dp.type_ == clrs.Type.POINTER:
                pointer_arr = dp.data[0] # (N,)
                edge_mask = (data_dict["edge_index"][1].numpy() == pointer_arr[data_dict["edge_index"][0].numpy()]).astype(float)
                data_dict[dp.name] = infer_type(dp.type_, edge_mask)
            else:
                data_dict[dp.name] = infer_type(dp.type_, dp.data[0])
            input_attributes.append(dp.name)
        else: # Graph
            data_dict[dp.name] = infer_type(dp.type_, dp.data[0])
            
    # Parse outputs
    for dp in outputs:
        output_attributes.append(dp.name)
        if dp.location == clrs.Location.EDGE:
            data_dict[dp.name] = infer_type(dp.type_, dp.data[0][data_dict["edge_index"][0], data_dict["edge_index"][1]])
        elif dp.location == clrs.Location.NODE:
            if dp.type_ == clrs.Type.POINTER:
                pointer_arr = dp.data[0] # (N,)
                edge_mask = (data_dict["edge_index"][1].numpy() == pointer_arr[data_dict["edge_index"][0].numpy()]).astype(float)
                data_dict[dp.name] = infer_type(dp.type_, edge_mask)
            else:
                data_dict[dp.name] = infer_type(dp.type_, dp.data[0])
        else: # Graph
            data_dict[dp.name] = infer_type(dp.type_, dp.data[0])
            
    if use_hints:
        # Parse hints
        for dp in hints:
            hint_attributes.append(dp.name)
            if dp.location == clrs.Location.EDGE or (dp.location == clrs.Location.NODE and dp.type_ == clrs.Type.POINTER):
                arr = dp.data.squeeze(1) # Hints, N, N or Hints, N
                if dp.location == clrs.Location.NODE: # Pointer
                    src = data_dict["edge_index"][0].numpy()
                    tgt = data_dict["edge_index"][1].numpy()
                    masks = (arr[:, src] == tgt[None, :]).astype(float).T
                    data_dict[dp.name] = infer_type(dp.type_, masks)
                else: # Edge
                    # Provide sparse conversion explicitly avoiding NxN memory bloat
                    num_dims = arr.ndim
                    transpose_indices = tuple(range(num_dims))
                    transpose_indices = (1, 2, 0) + transpose_indices[3:]
                    data_dict[dp.name] = infer_type(dp.type_, arr.transpose(*transpose_indices)[data_dict["edge_index"][0].numpy(), data_dict["edge_index"][1].numpy()])
            elif dp.location == clrs.Location.NODE and not dp.type_ == clrs.Type.POINTER:
                arr = dp.data.squeeze(1) # Hints, N, D (...)
                num_dims = arr.ndim
                transpose_indices = tuple(range(num_dims))
                transpose_indices = (1, 0) + transpose_indices[2:]
                data_dict[dp.name] = infer_type(dp.type_, arr.transpose(*transpose_indices))
            else:
                data_dict[dp.name] = infer_type(dp.type_, dp.data.squeeze(1)[np.newaxis, ...])

    data_dict = {k: to_torch(v) for k,v in data_dict.items()}
    data = CLRSData(**data_dict)    
    data.hints = hint_attributes
    data.inputs = input_attributes
    data.outputs = output_attributes
    return data

import salsaclrs.data
salsaclrs.data.to_sparse_data = efficient_to_sparse_data

import salsaclrs.sampler
from salsaclrs.sampler import BfsSampler
original_bfs_sampler_next = BfsSampler.next

def fast_bfs_sampler_next(self):
    import networkx as nx
    import scipy.sparse as sp
    import clrs
    import numpy as np
    from tests.utils.arxiv_loader import arxiv_graph_generator
    
    if self._graph_generator == "arxiv":
        generator_kwargs = self._get_graph_generator_kwargs()
        n = self._select_parameter(generator_kwargs.get('n', 0))
        G = arxiv_graph_generator(n, seed=None, return_type="networkx")
        
        source_node = self._rng.choice(G.number_of_nodes())
        
        # Raw adjacency (no self-loops) - used for the 'A' input
        A_raw = nx.to_scipy_sparse_array(G, dtype=float)
        # adj = probing.graph(A) = ((A + eye) != 0) * 1.0 — includes self-loops
        A_adj = A_raw + sp.eye(G.number_of_nodes(), dtype=float)
        
        # O(V + E) custom sparse BFS that precisely mimics dm-clrs matrix tie-breaking rules
        pi = np.arange(G.number_of_nodes(), dtype=np.int32)
        reach = np.zeros(G.number_of_nodes(), dtype=np.int8)
        reach[source_node] = 1
        
        A_csr = sp.csr_matrix(A_raw)
        frontier = [source_node]
        
        while frontier:
            # Sort the frontier to mathematically guarantee lowest-index node tie-breakers win
            frontier.sort()
            new_frontier = set()
            for i in frontier:
                start_ptr = A_csr.indptr[i]
                end_ptr = A_csr.indptr[i+1]
                neighbors = A_csr.indices[start_ptr:end_ptr]
                
                for j in neighbors:
                    if pi[j] == j and j != source_node:
                        pi[j] = i
                    if reach[j] == 0:
                        reach[j] = 1
                        new_frontier.add(j)
            frontier = list(new_frontier)
        
        inputs = [
            clrs.DataPoint(name="pos", location=clrs.Location.NODE, type_=clrs.Type.SCALAR, data=np.expand_dims(np.arange(G.number_of_nodes()) / G.number_of_nodes(), 0)),
            clrs.DataPoint(name="s", location=clrs.Location.NODE, type_=clrs.Type.MASK_ONE, data=np.expand_dims(np.arange(G.number_of_nodes()) == source_node, 0).astype(float)),
            clrs.DataPoint(name="A", location=clrs.Location.EDGE, type_=clrs.Type.SCALAR, data=[A_raw]),
            clrs.DataPoint(name="adj", location=clrs.Location.EDGE, type_=clrs.Type.MASK, data=[A_adj])
        ]
        
        outputs = [
            clrs.DataPoint(name="pi", location=clrs.Location.NODE, type_=clrs.Type.POINTER, data=np.expand_dims(pi, 0))
        ]
        
        # Empty hint list to avoid memory/time overhead of NxN algorithmic frames
        return inputs, outputs, []
    else:
        return original_bfs_sampler_next(self)

BfsSampler.next = fast_bfs_sampler_next
# ==============================================================================


from src.models.module import SALSACLRSModel
from src.utils.graph_generation import get_dataset
from salsaclrs import SALSACLRSDataModule
from salsaclrs.sampler import Sampler
from loguru import logger
from tests.utils.arxiv_loader import arxiv_graph_generator

# Import the format_results_table function from eval_checkpoint
from tests.eval_checkpoint import format_results_table
from tests.bfs_depth_analysis import analyse_batch, print_report

# Monkeypatch Sampler to support our Arxiv generator
from src.utils.graph_generation import patched_create_graph as existing_patched_create

def arxiv_aware_create_graph(self, n, weighted, directed, low=0.0, high=1.0, **kwargs):
    if self._graph_generator == "arxiv":
        n_val = self._select_parameter(n)
        return arxiv_graph_generator(n_val, seed=None)
    else:
        return existing_patched_create(self, n, weighted, directed, low=low, high=high, **kwargs)

Sampler._create_graph = arxiv_aware_create_graph

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of dataloader workers")
    parser.add_argument("--num-samples", type=int, default=1, help="Samples to generate per magnitude")
    parser.add_argument("--fast-iterations", action="store_true",
                        help="Use actual BFS depth instead of worst-case N iterations. "
                             "Faster but less methodologically correct.")
    parser.add_argument("--small", action="store_true", 
                        help="Limit evaluation to graphs with <= 16000 nodes (fast on CPU).")
    args = parser.parse_args()

    global USE_FAST_ITERATIONS
    USE_FAST_ITERATIONS = args.fast_iterations

    pl.seed_everything(args.seed)
    
    # Configure loguru to suppress excessive debug prints
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    # Ensure precision is set right
    torch.set_float32_matmul_precision('medium')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load Model
    logger.info(f"Loading checkpoint {args.ckpt} with strict=False for backward compatibility")
    model = SALSACLRSModel.load_from_checkpoint(args.ckpt, map_location=device, strict=False)
    
    # Get cfg directly from the loaded model
    cfg = model.cfg

    # Override TEST datasets to run Arxiv with large magnitude subgraphs
    cfg.DATA.TEST.NUM_SAMPLES = args.num_samples
    cfg.DATA.TEST.GRAPH_GENERATOR = ["arxiv", "arxiv", "arxiv", "arxiv", "arxiv"]
    cfg.DATA.TEST.NICKNAME = ["arxiv_16", "arxiv_160", "arxiv_1600", "arxiv_16000", "arxiv_169343"]
    cfg.DATA.TEST.GENERATOR_PARAMS = [
        {"n": 16, "directed": False, "acyclic": False, "weighted": False},
        {"n": 160, "directed": False, "acyclic": False, "weighted": False},
        {"n": 1600, "directed": False, "acyclic": False, "weighted": False},
        {"n": 16000, "directed": False, "acyclic": False, "weighted": False},
        {"n": 169343, "directed": False, "acyclic": False, "weighted": False} # The whole graph
    ]
    # Create a list of datasets to test on
    test_datasets = [
        {"n": 16, "directed": False, "acyclic": False, "weighted": False}, 
        {"n": 160, "directed": False, "acyclic": False, "weighted": False}, 
        {"n": 1600, "directed": False, "acyclic": False, "weighted": False}, 
        {"n": 16000, "directed": False, "acyclic": False, "weighted": False}, 
        {"n": 169343, "directed": False, "acyclic": False, "weighted": False}
    ]

    if args.small:
        logger.info("Limiting evaluation to small graphs (<= 16000 nodes) as requested by --small")
        cfg.DATA.TEST.GRAPH_GENERATOR = cfg.DATA.TEST.GRAPH_GENERATOR[:4]
        cfg.DATA.TEST.NICKNAME = cfg.DATA.TEST.NICKNAME[:4]
        cfg.DATA.TEST.GENERATOR_PARAMS = cfg.DATA.TEST.GENERATOR_PARAMS[:4]
        test_datasets = test_datasets[:4]

    logger.info("Loading test datasets (OGBN-Arxiv subgraphs)...")
    test_datasets_dict = get_dataset("test", cfg, seed=args.seed)
    
    datamodule = SALSACLRSDataModule(
        train_dataset=None,  # Not needed for testing
        val_datasets=[],     # Not needed for testing
        test_datasets=list(test_datasets_dict.values()), 
        batch_size=cfg.TRAIN.BATCH_SIZE, 
        num_workers=args.num_workers, 
        test_batch_size=cfg.TEST.BATCH_SIZE
    )

    # Monkeypatch for Kaggle/zero workers issues
    if args.num_workers == 0:
        _orig_dataloader = datamodule.dataloader
        def _patched_dataloader(dataset, **kwargs):
            kwargs["persistent_workers"] = False
            return _orig_dataloader(dataset, **kwargs)
        datamodule.dataloader = _patched_dataloader

    # Init Trainer
    trainer = pl.Trainer(
        accelerator="auto",
        logger=False, # Disable wandb logging for pure eval
        precision=cfg.TRAIN.PRECISION,
    )

    # Run Eval
    logger.info("Running standard evaluation on OGBN-Arxiv subgraphs...")
    results = trainer.test(model, datamodule=datamodule)

    # Print Table
    print("\n" + "="*80)
    print("ARXIV EVALUATION RESULTS")
    print("="*80)
    table = format_results_table(results)
    print(table)
    print("="*80 + "\n")

    # Run depth/reachability analysis similar to tests/bfs_depth_analysis.py
    logger.info("Running depth analysis and extracting statistics...")
    
    specs = model.specs
    output_key = None
    for k, v in specs.items():
        stage = v[0]
        stage_name = stage.name if hasattr(stage, "name") else str(stage)
        if stage_name.upper() == "OUTPUT":
            output_key = k
            break

    if output_key is not None:
        model.eval()
        model.to(device)
        for idx, loader in enumerate(datamodule.test_dataloader()):
            name = datamodule.get_test_loader_nickname(idx)
            all_results = []
            with torch.no_grad():
                for batch in loader:
                    batch = batch.to(device)
                    output, hints, hidden = model(batch)
                    all_results.extend(analyse_batch(batch, output, output_key, device))
            
            print_report(name, all_results)
    else:
        logger.warning("Could not find OUTPUT key in model specs; skipping depth analysis.")

if __name__ == '__main__':
    main()
