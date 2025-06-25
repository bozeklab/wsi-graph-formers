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
import os
from tqdm import tqdm
import glob

from utils.graph_utils import _is_numeric, _flatten


def nx_to_pyg_data(G: nx.Graph):
    """
    Convert a NetworkX graph into a PyTorch-Geometric ``Data`` object.

    Builds a feature matrix from **all numeric** node attributes and extracts
    node labels from the ``'cell_type'`` attribute.

    Notes
    -----
    * Every node is expected to expose *scalar or sequence* numeric attributes
      such as ``'area'``, ``'eccentricity'``, ``'centroid'``, …
    * Attribute names are sorted alphabetically to guarantee a deterministic
      feature order.
    * Missing attributes are imputed with ``0.0`` so that every node ends up
      with the same feature dimensionality.
    * The original ``networkx`` graph remains unchanged.

    Parameters
    ----------
    G : networkx.Graph
        Input graph. Each node **must** have
        * one or more numeric attributes (int, float, list, tuple, ndarray)
        * an optional ``'cell_type'`` label (int, default = ``-1``).

    Returns
    -------
    data : torch_geometric.data.Data
        PyG ``Data`` object with:
        * ``x`` ― ``FloatTensor`` of shape ``[num_nodes, num_features]``
        * ``y`` ― ``LongTensor``  of shape ``[num_nodes]``
        * ``edge_index`` ― ``LongTensor`` of shape ``[2, num_edges]``
    """

    # Determine which node-attribute keys will form the feature vector 
    _, first_attr = next(iter(G.nodes(data=True)))
    feature_keys = sorted(
        k for k, v in first_attr.items()
        if k != "cell_type" and _is_numeric(v)
    )

    # Size of the flattened feature vector per node
    feat_dim = sum(_flatten(first_attr[k]).size for k in feature_keys)

    # Collect per-node feature and label tensors                         
    xs, ys = [], []
    for _, attr in G.nodes(data=True):
        parts = []
        for k in feature_keys:
            parts.append(_flatten(attr.get(k, 0.0)))
        xs.append(np.concatenate(parts).astype(np.float32))
        ys.append(int(attr.get("cell_type", -1)))

    x = torch.tensor(np.stack(xs), dtype=torch.float)     # [N, F]
    y = torch.tensor(ys, dtype=torch.long)                # [N]

    # Let PyG build edge_index (and optional edge_attr)                  
    # edge_index etc.
    data = from_networkx(G)  
    data.x, data.y = x, y

    return data




@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):

    # in case cfg.conversion_output_folder does not exiqsts 
    if not os.path.exists(cfg.conversion_output_folder):
        os.mkdir(cfg.conversion_output_folder)

    outputext = '.pt'
    picklepaths = os.path.join(cfg.pickle_output_folder, '*.pickle')
    picklepaths = glob.glob(picklepaths) 
    for filepath in tqdm(picklepaths):
        if os.path.exists(filepath):
            pickle_path = filepath
            filename = str(os.path.splitext(os.path.split(filepath)[1])[0])
            output_path = cfg.conversion_output_folder + filename + outputext

            with open(pickle_path, "rb") as f:
                G = pickle.load(f)
            pyg_graph = nx_to_pyg_data(G)
            # we do not writte a function for saving as it is basically one line and one print
            torch.save(pyg_graph, output_path)
            print(f"Converted graph saved to: {output_path}")

    print("Conversion complete!")

if __name__ == "__main__":
    main()
