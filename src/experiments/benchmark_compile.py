import os
import sys
import time
import torch
import gc

# Add project root to sys.path so absolute imports work when running directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.models.module import SALSACLRSModel
from src.utils.config import load_cfg
from salsaclrs import SALSACLRSDataModule
from src.utils.graph_generation import get_dataset

def benchmark_compile(compile_mode: str, cfg_path: str, num_batches: int = 50):
    print(f"\n{'='*50}")
    print(f"Benchmarking with compile_mode = '{compile_mode}'")
    print(f"{'='*50}")

    # Load config and override compilation setting
    cfg = load_cfg(cfg_path)

    try:
        # Load datasets (using train set for benchmarking)
        print("Loading dataset...", end="", flush=True)
        train_ds = get_dataset("train", cfg)
        val_ds = get_dataset("val", cfg)
        test_datasets = get_dataset("test", cfg)
        
        datamodule = SALSACLRSDataModule(
            train_dataset=train_ds,
            val_datasets=list(val_ds.values()), 
            test_datasets=list(test_datasets.values()), 
            batch_size=cfg.TRAIN.BATCH_SIZE, 
            num_workers=cfg.TRAIN.NUM_WORKERS, 
            test_batch_size=cfg.TEST.BATCH_SIZE
        )
        datamodule.setup('fit')
        dl = datamodule.train_dataloader()
        dl_iter = iter(dl)
        print(" Done.")

        print(f"Initializing model with compile_mode '{compile_mode}'...", end="", flush=True)
        model = SALSACLRSModel(specs=train_ds.specs, cfg=cfg, compile_mode=compile_mode)
        model = model.to('cuda')
        model.train()
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        print(" Done.")

        # Warmup (5 batches to trigger compilations if enabled)
        print("Starting warmup (5 batches)...")
        for i in range(5):
            batch = next(dl_iter).to('cuda')
            optimizer.zero_grad()
            loss = model.training_step(batch, batch_idx=i)
            loss.backward()
            optimizer.step()
            
        print("Warmup complete. Recompilations should be mostly done.")

        # Benchmark
        print(f"Benchmarking {num_batches} batches...")
        torch.cuda.synchronize()
        start_time = time.time()
        
        for i in range(num_batches):
            try:
                batch = next(dl_iter)
            except StopIteration:
                dl_iter = iter(dl)
                batch = next(dl_iter)
                
            batch = batch.to('cuda')
            optimizer.zero_grad()
            loss = model.training_step(batch, batch_idx=i)
            loss.backward()
            optimizer.step()
            
        torch.cuda.synchronize()
        end_time = time.time()
        
        total_time = end_time - start_time
        it_per_sec = num_batches / total_time
        
        print(f"--- Results ---")
        print(f"Total time for {num_batches} batches: {total_time:.3f} s")
        print(f"Throughput: {it_per_sec:.2f} it/s")
        print(f"{'='*50}\n")
        
    finally:
        # Cleanup memory
        del model, optimizer, train_ds, datamodule
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="src/configs/bfs/PGN-grokking.yml")
    args = parser.parse_args()
    
    # Ensure TF32 is enabled
    torch.set_float32_matmul_precision('high')

    # Run Benchmark across all three modes
    benchmark_compile(compile_mode="none", cfg_path=args.cfg)
    benchmark_compile(compile_mode="full", cfg_path=args.cfg)
    benchmark_compile(compile_mode="surgical", cfg_path=args.cfg)
    
