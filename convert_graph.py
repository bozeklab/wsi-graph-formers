"""
Lucas Sancéré 2025
"""

import pickle
import torch
import numpy as np
import networkx as nx
from torch_geometric.utils import from_networkx
import hydra
from omegaconf import DictConfig
from pathlib import Path

def nx_to_pyg_data(G: nx.Graph) -> torch.Tensor:
    """
    Convert a NetworkX graph into a PyTorch‑Geometric Data object.

    Extracts node features and labels from node attributes, wraps edges.

    Parameters
    ----------
    G : networkx.Graph
        Graph where each node has:
        - 'features': array-like, the node’s feature vector
        - 'cell_type': int or label (optional, default = -1)
    Returns
    -------
    data : torch_geometric.data.Data
        PyG Data object with attributes:
        - x: FloatTensor of shape [num_nodes, num_features]
        - y: LongTensor of shape [num_nodes]
        - edge_index: LongTensor of shape [2, num_edges]
    """

    # ensure every node has x (features) and y (label)
    first = next(iter(G.nodes(data=True)))[1]
    feat_dim = len(first['features'])
    for node, attr in G.nodes(data=True):
        G.nodes[node]['x'] = np.array(attr['features'], dtype=np.float32)
        G.nodes[node]['y'] = int(attr.get('cell_type', -1))
    data = from_networkx(G, group_node_attrs=['x', 'y'])
    data.x = data.x.view(-1, feat_dim)
    data.y = data.y.view(-1).long()
    return data

@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    in_path = cfg.conversion_input_folder
    out_path = cfg.conversion_output_folder
    print(f"Converting {in_path} → {out_path}")

    with open(in_path, "rb") as f:
        G = pickle.load(f)

    data = nx_to_pyg_data(G)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, out_path)
    print("Conversion complete!")

if __name__ == "__main__":
    main()
