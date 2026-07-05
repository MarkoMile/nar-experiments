import os, sys
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch, torch_scatter, numpy as np, networkx as nx
from pyvis.network import Network
from IPython.display import IFrame, display, HTML
import pandas as pd
import ipywidgets as widgets
import tkinter as tk
from tkinter import filedialog
from loguru import logger

from src.utils.graph_generation import get_dataset
from src.models.module import SALSACLRSModel, calc_metrics

def fix_batch_attributes(batch):
    for attr in ("inputs", "outputs", "hints"):
        val = getattr(batch, attr, None)
        if isinstance(val, list) and val and isinstance(val[0], list):
            setattr(batch, attr, val[0])
    batch.inputs  = ["pos", "s"]
    batch.outputs = ["pi"]
    batch.hints   = ["reach_h", "pi_h"]
    return batch

def pred_parent_per_node(pred_pi, edge_index, num_nodes):
    _, argmax_edge = torch_scatter.scatter_max(
        pred_pi, edge_index[0], dim=-1, dim_size=num_nodes
    )
    valid = argmax_edge < edge_index.shape[1]
    parent = torch.arange(num_nodes)
    parent[valid] = edge_index[1, argmax_edge[valid]]
    return parent

def true_parent_per_node(pi, edge_index, num_nodes):
    _, argmax_edge = torch_scatter.scatter_max(
        pi.float(), edge_index[0], dim=-1, dim_size=num_nodes
    )
    valid = argmax_edge < edge_index.shape[1]
    parent = torch.arange(num_nodes)
    parent[valid] = edge_index[1, argmax_edge[valid]]
    return parent

def build_undirected_graph(edge_index, num_nodes):
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    seen = set()
    for i in range(edge_index.shape[1]):
        u, v = int(edge_index[0, i]), int(edge_index[1, i])
        key = (min(u, v), max(u, v))
        if key not in seen:
            seen.add(key)
            G.add_edge(u, v)
    return G

def draw_comparison_graph(data, pred_pi, title=None, height="800px"):
    num_nodes = data.s.shape[0]
    source    = data.s.argmax().item()
    G         = build_undirected_graph(data.edge_index, num_nodes)

    pred_par  = pred_parent_per_node(pred_pi, data.edge_index, num_nodes).numpy()
    true_par  = true_parent_per_node(data.pi,  data.edge_index, num_nodes).numpy()

    pos = nx.spring_layout(G, seed=42, k=10.0 / np.sqrt(num_nodes))

    reachable   = [n for n in G.nodes() if n != source and true_par[n] != n]
    unreachable = [n for n in G.nodes() if n != source and true_par[n] == n]

    net = Network(notebook=True, width="100%", height=height, directed=True, cdn_resources='in_line')
    
    for n in G.nodes():
        if n == source:
            color = "#f4a261"
            shape = "star"
            size = 25
            label = f"{n} (Source)"
        elif n in unreachable:
            color = "#bbbbbb"
            shape = "dot"
            size = 10
            label = str(n)
        else:
            color = "#457b9d"
            shape = "dot"
            size = 15
            label = str(n)
            
        x, y = pos[n]
        net.add_node(int(n), label=label, color=color, shape=shape, size=size, x=float(x)*1000, y=float(y)*1000)
        
    for u, v in G.edges():
        net.add_edge(int(u), int(v), color="#d0d0d0", width=1, arrows="")

    for n in range(num_nodes):
        if n == source:
            continue
        pp = pred_par[n]
        tp = true_par[n]
        if pp != n:
            if pp == tp:
                net.add_edge(int(n), int(pp), color="#2a9d8f", width=3, arrows="to")
            else:
                net.add_edge(int(n), int(pp), color="#e63946", width=3, arrows="to")
        if tp != n and pp != tp:
            net.add_edge(int(n), int(tp), color="#6c757d", width=2, arrows="to", dashes=True)

    net.force_atlas_2based(
        gravity=-50,
        central_gravity=0.01,
        spring_length=100,
        spring_strength=0.08,
        damping=0.4,
        overlap=1.0
    )
    net.show_buttons(filter_=['physics'])
    
    filename = "bfs_visualization.html"
    net.show(filename)
    
    if title:
        display(HTML(f"<h3>{title}</h3>"))

    custom_ui = """
<div style="position: absolute; top: 10px; left: 10px; z-index: 9999; background: white; padding: 15px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); font-family: sans-serif; min-width: 200px; border: 1px solid #ccc;">
  <div style="margin-bottom: 10px;">
    <b>Prediction Edge Thickness:</b> <span id="customEdgeLabel">3.0</span><br>
    <input type="range" id="customEdgeSlider" min="0.5" max="10.0" step="0.5" value="3.0" style="width: 100%;">
  </div>
  <div>
    <label style="cursor: pointer; display: flex; align-items: center;">
      <input type="checkbox" id="customBaseEdgesToggle" checked style="margin-right: 8px;">
      <b>Show Base Graph Edges</b>
    </label>
  </div>
</div>
<script>
var initCustomUI = setInterval(function() {
  if (typeof edges !== 'undefined' && edges !== null) {
    clearInterval(initCustomUI);

    var slider = document.getElementById('customEdgeSlider');
    var label = document.getElementById('customEdgeLabel');
    var toggle = document.getElementById('customBaseEdgesToggle');

    var baseWidths = {};
    edges.get().forEach(function(e) {
      baseWidths[e.id] = e.width || 1;
    });

    function updateEdges() {
      var wMultiplier = parseFloat(slider.value) / 3.0;
      label.innerText = parseFloat(slider.value).toFixed(1);
      var showBase = toggle.checked;

      var updates = edges.get().map(function(e) {
         var isBase = (e.color === "#d0d0d0") || (e.color && e.color.color === "#d0d0d0");
         var newHidden = isBase ? !showBase : false;
         
         var newWidth = baseWidths[e.id];
         if (!isBase && baseWidths[e.id] > 1) {
             newWidth = baseWidths[e.id] * wMultiplier;
         }

         return { id: e.id, hidden: newHidden, width: newWidth };
      });
      edges.update(updates);
    }

    slider.addEventListener('input', updateEdges);
    toggle.addEventListener('change', updateEdges);
  }
}, 500);
</script>
"""
    with open(filename, "r") as f:
        html_content = f.read()
    
    if "</body>" in html_content:
        html_content = html_content.replace("</body>", custom_ui + "\n</body>")
    else:
        html_content += custom_ui
    import html as _html_lib
    escaped_html = _html_lib.escape(html_content)
    display(HTML(f'<iframe srcdoc="{escaped_html}" width="100%" height="{height}" style="border:none;"></iframe>'))

def run_evaluation_and_visualization():
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    print(f"Opening pop-up dialog to select checkpoint...")
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    initial_dir = os.path.join(project_root, "model-checkpoints")
    if not os.path.exists(initial_dir):
        initial_dir = project_root

    CHECKPOINT_PATH = filedialog.askopenfilename(
        parent=root,
        title="Select a Checkpoint File",
        initialdir=initial_dir,
        filetypes=[("Checkpoint files", "*.ckpt"), ("All files", "*.*")]
    )
    root.destroy()

    if not CHECKPOINT_PATH:
        raise RuntimeError("No checkpoint selected. Halting execution.")
        
    print(f"✅ Successfully loaded checkpoint: {os.path.relpath(CHECKPOINT_PATH, project_root)}")

    model = SALSACLRSModel.load_from_checkpoint(CHECKPOINT_PATH, map_location="cpu", strict=False)
    cfg = model.cfg
    cfg.DATA.ROOT = os.path.join(project_root, "data")
    model.eval()

    test_datasets = get_dataset("test", cfg)

    from torch_geometric.data import Batch
    
    target_set = "ws_80"
    if target_set in test_datasets:
        ds = test_datasets[target_set]
        sample = ds[0]
        batch = Batch.from_data_list([sample])
        batch = fix_batch_attributes(batch)
        
        print(f"✅ Generating BFS Rollout visualization for a single graph ({target_set}, N={sample.num_nodes}) to speed up execution...")
        
        with torch.no_grad():
            output, hints, hidden = model(batch)
            m = calc_metrics("pi", output, batch, model.specs["pi"][2])
            ga = m["graph_result"].float().mean().item()
            na = m["node_accuracy"].float().mean().item()
            
        print(f"📊 Accuracy on this graph -> Graph Match: {ga*100:.1f}% | Node Accuracy: {na*100:.2f}%")
            
        draw_comparison_graph(
            sample, output["pi"],
            title=f"{target_set} (N={sample.num_nodes})"
        )
    else:
        print(f"Dataset {target_set} not found in test_datasets. Available keys: {list(test_datasets.keys())}")
