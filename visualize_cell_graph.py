"""
Lucas Sancéré 2025
"""

import networkx as nx
import matplotlib.pyplot as plt
import random
from omegaconf import DictConfig
import hydra
from hydra.core.config_store import ConfigStore
from configs.schema import GraphConfig
import pickle

cs = ConfigStore.instance()
cs.store(name="graph_config", node=GraphConfig)


def plot_graph(G, max_nodes=1000, figsize=(10, 10)):
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
    """
    num_nodes = G.number_of_nodes()

    if num_nodes == 0:
        print("Warning: Graph is empty. Nothing to visualize.")
        return

    if num_nodes > max_nodes:
        print(f"Graph has {num_nodes} nodes. Sampling {max_nodes}.")

    # Get largest connected component
    largest_cc = max(nx.connected_components(G), key=len)
    if len(largest_cc) > max_nodes:
        sampled_nodes = random.sample(list(largest_cc), max_nodes)
    else:
        sampled_nodes = list(largest_cc)

    G = G.subgraph(sampled_nodes).copy()


    # Get node positions (based on centroids) and colors (cell type)
    pos = {n: G.nodes[n].get("centroid", (0, 0)) for n in G.nodes}
    colors = [G.nodes[n].get("cell_type", 0) for n in G.nodes]

    plt.figure(figsize=figsize)
    nx.draw(
        G,
        pos,
        node_color=colors,
        node_size=10,
        cmap="tab10",
        edge_color="gray",
        linewidths=0.1,
    )
    plt.title("Cell Graph (colored by cell type)")
    plt.axis("equal")
    plt.tight_layout()
    plt.show()


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    with open(cfg.output_path, "rb") as f:
        G = pickle.load(f)

    figsize = tuple(cfg.figsize)
    # Validate figsize to avoid accidental 1000x1000 inches
    figsize = tuple(min(max(float(x), 2), 20) for x in figsize)

    # check: Are centroids present?
    if not all("centroid" in G.nodes[n] for n in G.nodes):
        print("Warning: Some nodes are missing 'centroid'. Falling back to spring_layout.")
        pos = nx.spring_layout(G, seed=42)
    else:
        pos = {n: G.nodes[n]["centroid"] for n in G.nodes}


    plot_graph(G, max_nodes=cfg.max_nodes_display, figsize=figsize)


if __name__ == "__main__":
    main()
