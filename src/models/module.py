"""
PyTorch Lightning module for training and evaluating GNN models on SALSA-CLRS algorithmic reasoning tasks.

Wraps the EncodeProcessDecode model with multi-component loss computation (output, hint, hidden),
per-batch metric calculation, epoch-level metric aggregation, and configurable optimizer/scheduler setup.
"""
import lightning.pytorch as pl
import torch.nn as nn
import torch.nn.functional as F
import torch
from collections import defaultdict
from loguru import logger
from sklearn.metrics import accuracy_score, precision_score, recall_score

from src.models.models import EncodeProcessDecode
from src.models.loss import CLRSLoss
from src.utils.utils import stack_dicts

from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau


def calc_metrics(key, preds, batch, type_):
    truth = batch[key]
    preds = preds[key]
    # count ones
    graph_sizes = [(batch.batch == i).sum() for i in range(0, max(batch.batch)+1)]

    if type_ == "pointer":
        """Node Acc. and Graph Acc."""
        node_level = [[] for _ in range(batch.num_graphs)]
        
        for n in range(batch.num_nodes):
            idx = (batch.edge_index[0] == n)
            graph_idx = batch.batch[n]
            
            correct = (preds[idx].argmax(dim=-1) == truth[idx].argmax(dim=-1)).float()
            
            # Calculate node metrics
            node_level[graph_idx].append(correct)

        # Collapse to per graph metrics
        node_level = [torch.tensor(x) for x in node_level]

        # Graph Metrics
        graph_result = torch.tensor([x.all() for x in node_level])

        # Node Metrics
        node_acc = torch.tensor([x.mean() for x in node_level])

        return {
            "node_accuracy": node_acc,
            "graph_result": graph_result,
        }

    elif type_ == "mask":
        """Node Acc., Graph Acc."""
        if truth.sum() < 0.05 * truth.numel():
            logger.warning(f"MASK METRIC: Truth has less than 5% ones: {truth.sum()} / {truth.numel()}")

        preds = preds.sigmoid()
        node_acc = []
        graph_result = []


        for n in range(batch.num_graphs):
            gpred = (preds[batch.batch == n]>0.5).bool().cpu().numpy()
            gtruth = truth[batch.batch == n].cpu().numpy()

            node_acc.append(accuracy_score(gtruth, gpred))
            graph_result.append((gpred == gtruth).all())

        graph_result = torch.tensor(graph_result)

        return {
                "node_accuracy": node_acc,
                "graph_result": graph_result,
            }

    elif type_ == "scalar":
        """MSE, Graph Acc."""

        mse = []
        graph_result = []

        for n in range(batch.num_graphs):
            gpred = preds[n]
            gtruth = truth[n]

            # Use double to avoid float32 squaring overflow during evaluation
            mse.append(((gpred.double() - gtruth.double())**2).mean().float())
            graph_result.append((gpred.round() == gtruth).all().float() if gpred.dim() > 0 else (gpred.round() == gtruth).float())

        mse = torch.tensor(mse)
        graph_result = torch.tensor(graph_result)

        return {
                "mse": mse,
                "graph_result": graph_result,
            }
    else:
        raise NotImplementedError(f"Unknown metric type {type_}")     

class SALSACLRSModel(pl.LightningModule):
    def __init__(self, specs, cfg):
        super().__init__()
        self.hparams.update(cfg)
        self.cfg = cfg
        self.model = EncodeProcessDecode(specs, cfg)
        self.loss = CLRSLoss(specs, cfg.TRAIN.LOSS.HIDDEN_LOSS_TYPE)
        self.step_output_cache = defaultdict(list)
        self.current_loader_idx = 0
        self.specs = specs
        self.save_hyperparameters()

    def forward(self, batch):
        return self.model(batch)        

    def _loss(self, batch, output, hints, hidden):
        outloss, hintloss, hiddenloss = self.loss(batch, output, hints, hidden)
        loss = self.cfg.TRAIN.LOSS.OUTPUT_LOSS_WEIGHT * outloss + self.cfg.TRAIN.LOSS.HINT_LOSS_WEIGHT * hintloss + self.cfg.TRAIN.LOSS.HIDDEN_LOSS_WEIGHT * hiddenloss
        return loss, outloss, hintloss, hiddenloss

    def training_step(self, batch, batch_idx):
        output, hints, hidden = self.model(batch)
        loss, outloss, hintloss, hiddenloss = self._loss(batch, output, hints, hidden)
        self.log("train/outloss", outloss, batch_size=batch.num_graphs)
        self.log("train/hintloss", hintloss, batch_size=batch.num_graphs)
        self.log("train/hiddenloss", hiddenloss, batch_size=batch.num_graphs)
        self.log("train/loss", loss, batch_size=batch.num_graphs)
        self.log('train/lr', self.trainer.optimizers[0].param_groups[0]['lr'])

        # Log total weight norm to track grokking phase transition
        # (weights inflate during memorization, deflate when weight decay kicks in)
        if self.global_step % 50 == 0:
            with torch.no_grad(): # Ensure this doesn't build a computation graph!
                total_norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(p.double(), 2) for p in self.parameters() if p.requires_grad]), 2).float()
                self.log('train/weight_norm', total_norm, batch_size=batch.num_graphs)

        return loss

    def _shared_eval(self, batch, dataloader_idx, stage):
        output, hints, hidden = self.model(batch)
        loss, outloss, hintloss, hiddenloss = self._loss(batch, output, hints, hidden)
        self.current_loader_idx = dataloader_idx
        # calc batch metrics
        assert len(batch.outputs) == 1
        metrics = calc_metrics(batch.outputs[0], output, batch, self.specs[batch.outputs[0]][2])
        output.update({f"{m}_metric": metrics[m] for m in metrics})
        output["batch_size"] = torch.tensor(batch.num_graphs).float()
        output["num_nodes"] = torch.tensor(batch.num_nodes).float()
        return loss, outloss, hintloss, output

    def _end_of_epoch_metrics(self, dataloader_idx):
        output = stack_dicts(self.step_output_cache[dataloader_idx])
        # average metrics over graphs
        metrics = {}
        for m in output:
            if not m.endswith("_metric"):
                continue
            if m.startswith("graph"):
                # graph level metrics have to be computed differently
                metrics["graph_accuracy"] = output[m].float().mean()
            else:
                metrics[m[:-7]] = output[m].float().mean()
        return metrics
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss, outloss, hintloss, output = self._shared_eval(batch, dataloader_idx, "val")
        self.log(f'val/loss/{self.trainer.datamodule.get_val_loader_nickname(dataloader_idx)}', loss, batch_size=batch.num_graphs, add_dataloader_idx=False)
        self.log(f'val/outloss/{self.trainer.datamodule.get_val_loader_nickname(dataloader_idx)}', outloss, batch_size=batch.num_graphs, add_dataloader_idx=False)
        self.log(f'val/hintloss/{self.trainer.datamodule.get_val_loader_nickname(dataloader_idx)}', hintloss, batch_size=batch.num_graphs, add_dataloader_idx=False)

        self.step_output_cache[dataloader_idx].append(output)
        return loss
    
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        loss, outloss, hintloss, output = self._shared_eval(batch, dataloader_idx, "test")
        self.log(f'test/loss/{self.trainer.datamodule.get_test_loader_nickname(dataloader_idx)}', loss, batch_size=batch.num_graphs, add_dataloader_idx=False)
        self.log(f'test/outloss/{self.trainer.datamodule.get_test_loader_nickname(dataloader_idx)}', outloss, batch_size=batch.num_graphs, add_dataloader_idx=False)
        self.log(f'test/hintloss/{self.trainer.datamodule.get_test_loader_nickname(dataloader_idx)}', hintloss, batch_size=batch.num_graphs, add_dataloader_idx=False)

        self.step_output_cache[dataloader_idx].append(output)
        return loss
    
    def on_validation_epoch_end(self):
        for dataloader_idx in self.step_output_cache.keys():
            metrics = self._end_of_epoch_metrics(dataloader_idx)
            for key in metrics:
                self.log(f"val/{key}/{self.trainer.datamodule.get_val_loader_nickname(dataloader_idx)}", metrics[key], add_dataloader_idx=False)
        self.step_output_cache.clear()

    def on_test_epoch_end(self):
        for dataloader_idx in self.step_output_cache.keys():
            metrics = self._end_of_epoch_metrics(dataloader_idx)
            for key in metrics:
                self.log(f"test/{key}/{self.trainer.datamodule.get_test_loader_nickname(dataloader_idx)}", metrics[key], add_dataloader_idx=False)
        self.step_output_cache.clear()  

    def configure_optimizers(self):
        if self.cfg.TRAIN.OPTIMIZER.NAME == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.TRAIN.OPTIMIZER.LR)
        elif self.cfg.TRAIN.OPTIMIZER.NAME == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.TRAIN.OPTIMIZER.LR, weight_decay=self.cfg.TRAIN.OPTIMIZER.WEIGHT_DECAY)
        else:
            raise NotImplementedError(f"Optimizer {self.cfg.TRAIN.OPTIMIZER.NAME} not implemented")
        out = {"optimizer": optimizer}
        if self.cfg.TRAIN.SCHEDULER.ENABLE:
            try:
                scheduler_params = dict(self.cfg.TRAIN.SCHEDULER.PARAMS[0])
                if self.cfg.TRAIN.SCHEDULER.NAME == "ReduceLROnPlateau":
                    scheduler_params["mode"] = self.cfg.TRAIN.CHECKPOINT_MONITOR_MODE
                scheduler = getattr(torch.optim.lr_scheduler, self.cfg.TRAIN.SCHEDULER.NAME)(optimizer, **scheduler_params)
                if hasattr(self, 'trainer') and self.trainer is not None and hasattr(self.trainer, 'datamodule') and self.trainer.datamodule is not None:
                    nickname = self.trainer.datamodule.get_val_loader_nickname(0)
                    monitor_metric = self.cfg.TRAIN.CHECKPOINT_MONITOR.format(val_nickname=nickname)
                else:
                    monitor_metric = self.cfg.TRAIN.CHECKPOINT_MONITOR.format(val_nickname="0")

                out["lr_scheduler"] = {
                    "scheduler": scheduler,
                    "monitor": monitor_metric,
                    "interval": "epoch",
                    "frequency": 1,
                }
            except AttributeError:
                raise NotImplementedError(f"Scheduler {self.cfg.TRAIN.SCHEDULER.NAME} not implemented")

        return out