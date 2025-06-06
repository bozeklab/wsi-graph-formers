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
        Size of the matplotlib figure.
    """
    if G.number_of_nodes() > max_nodes:
        print(f"Graph has {G.number_of_nodes()} nodes. Sampling {max_nodes}.")
        nodes = random.sample(list(G.nodes), max_nodes)
        G = G.subgraph(nodes)

    pos = {n: G.nodes[n].get("centroid", (0, 0)) for n in G.nodes}
    types = nx.get_node_attributes(G, "cell_type")
    colors = [types.get(n, 0) for n in G.nodes]

    plt.figure(figsize=figsize)
    nx.draw(G, pos, node_color=colors, node_size=10, cmap='tab10', edge_color='gray', linewidths=0.1)
    plt.title("Cell Graph (colored by cell type)")
    plt.axis("equal")
    plt.show()


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    G = nx.read_gpickle(cfg.output_path)
    plot_graph(G, max_nodes=cfg.max_nodes_display, figsize=tuple(cfg.figsize))

if __name__ == "__main__":
    main()
