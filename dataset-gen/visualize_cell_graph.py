"""
Lucas Sancéré 2025
"""

import sys
sys.path.append('../')  # Only for Remote use on Cluste

import networkx as nx
import matplotlib.pyplot as plt
import random
from omegaconf import DictConfig
import hydra
from hydra.core.config_store import ConfigStore
from configs.schema import GraphConfig
import pickle
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data
import torch

# cs = ConfigStore.instance()
# cs.store(name="graph_config", node=GraphConfig)

COLORS = [
    (0, 0, 0),       # 0 - Background (white)  
    (1.0, 1.0, 0.0),       # 1 - Granulocyte (yellow)  
    (0.078, 0.914, 0.078), # 2 - Lymphocyte (green)
    (0.055, 0.949, 0.965), # 3 - Plasma (light blue)
    (0.063, 0.020, 0.945), # 4 - Stroma (dark blue)
    (1.0, 0.0039, 0.0),    # 5 - Tumor (red)  
    (1.0, 0.690, 0.067)    # 6 - Epithelial  (orange)
]




def plot_nx_graph(G, max_nodes=1000, figsize=(10, 10), only_largest_cc=False):
    """
    Plot a graph using NetworkX with node colors representing cell types.

    Parameters
    ----------
    G : networkx.Graph
        The graph to be visualized.
    max_nodes : int
        Maximum number of nodes to visualize.
    figsize : tuple
        Size of the matplotlib figure in inches.
    only_largest_cc : bool
        If True, visualize only the largest connected component.
    """
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    if num_nodes == 0:
        print("Warning: Graph is empty. Nothing to visualize.")
        return

    if only_largest_cc:
        components = list(nx.connected_components(G))
        largest_cc = max(components, key=len)
        nodes_to_plot = (
            random.sample(list(largest_cc), max_nodes)
            if len(largest_cc) > max_nodes
            else list(largest_cc)
        )
        G = G.subgraph(nodes_to_plot).copy()
    else:
        if num_nodes > max_nodes:
            print(f"Graph has {num_nodes} nodes and {num_edges} edges. Sampling {max_nodes} nodes.")
            sampled_nodes = random.sample(list(G.nodes()), max_nodes)
            G = G.subgraph(sampled_nodes).copy()
        else:
            print(f"Graph has {num_nodes} nodes and {num_edges} edges. All nodes and edges are displayed.")

    pos = {n: G.nodes[n].get("centroid", (0, 0)) for n in G.nodes}
    colors = [
        COLORS[G.nodes[n].get("cell_type", 0)]
        if G.nodes[n].get("cell_type", 0) < len(COLORS)
        else COLORS[-1]
        for n in G.nodes
    ]

    plt.figure(figsize=figsize)
    nx.draw(
        G,
        pos,
        node_color=colors,
        node_size=10,
        edge_color="grey",
        linewidths=0.1,
    )
    plt.title("Cell Graph (colored by cell type)")
    plt.axis("equal")
    plt.tight_layout()
    plt.show()




def plot_pyg_graph(data, max_nodes=1000, figsize=(10, 10), only_largest_cc=False):
    """
    Plot a PyTorch-Geometric graph with node colors representing cell types.

    Parameters
    ----------
    data : torch_geometric.data.Data
        PyG Data object with edge_index, cell_type (y), and optionally 'centroid'.
    max_nodes : int
        Maximum number of nodes to visualize.
    figsize : tuple
        Size of the matplotlib figure in inches.
    only_largest_cc : bool
        If True, visualize only the largest connected component.
    """
    # Convert to NetworkX for visualization
    G = to_networkx(data, to_undirected=True)

    # Attach necessary node attributes
    for i in G.nodes:
        G.nodes[i]["cell_type"] = int(data.y[i].item()) if hasattr(data, "y") else 0
        if hasattr(data, "centroid"):
            G.nodes[i]["centroid"] = tuple(data.centroid[i].tolist())
        else:
            G.nodes[i]["centroid"] = (0, 0)  # fallback

    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    if num_nodes == 0:
        print("Warning: Graph is empty. Nothing to visualize.")
        return

    if only_largest_cc:
        components = list(nx.connected_components(G))
        largest_cc = max(components, key=len)
        nodes_to_plot = (
            random.sample(list(largest_cc), max_nodes)
            if len(largest_cc) > max_nodes
            else list(largest_cc)
        )
        G = G.subgraph(nodes_to_plot).copy()
    else:
        if num_nodes > max_nodes:
            print(f"Graph has {num_nodes} nodes and {num_edges} edges. Sampling {max_nodes} nodes.")
            sampled_nodes = random.sample(list(G.nodes()), max_nodes)
            G = G.subgraph(sampled_nodes).copy()
        else:
            print(f"Graph has {num_nodes} nodes and {num_edges} edges. All nodes and edges are displayed.")

    # Positions (from centroid or fallback to spring_layout)
    pos = {n: G.nodes[n].get("centroid", (0, 0)) for n in G.nodes}

    # Color by cell type
    colors = [
        COLORS[G.nodes[n].get("cell_type", 0)]
        if G.nodes[n].get("cell_type", 0) < len(COLORS)
        else COLORS[-1]
        for n in G.nodes
    ]

    plt.figure(figsize=figsize)
    nx.draw(
        G,
        pos,
        node_color=colors,
        node_size=10,
        edge_color="grey",
        linewidths=0.1,
    )
    plt.title("Cell Graph (colored by cell type)")
    plt.axis("equal")
    plt.tight_layout()
    plt.show()





@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):

    ## Figure parameterss
    figsize = tuple(cfg.figsize)
    # Validate figsize to avoid accidental 1000x1000 inches
    figsize = tuple(min(max(float(x), 2), 20) for x in figsize)

    ## Graph plot
    if cfg.graphtype == 'pyg':
        loaded_dict = torch.load(cfg.visualization_path)
        data = Data(**loaded_dict)  # <-- convert dict to PyG Data object
        plot_pyg_graph(
            data, 
            max_nodes=cfg.max_nodes_display, 
            figsize=figsize
            )
        

    elif  cfg.graphtype == 'nx':
        with open(cfg.visualization_path, "rb") as f:
            G = pickle.load(f)

        # check: Are centroids present?
        if not all("centroid" in G.nodes[n] for n in G.nodes):
            print("Warning: Some nodes are missing 'centroid'. Falling back to spring_layout.")
            pos = nx.spring_layout(G, seed=42)
        else:
            pos = {n: G.nodes[n]["centroid"] for n in G.nodes}
        
        plot_nx_graph(
            G, 
            max_nodes=cfg.max_nodes_display, 
            figsize=figsize
            )

    else:
        raise ValueError('cfg.graphtype is not defined properly. It has to be nx or pyg')




if __name__ == "__main__":
    main()
