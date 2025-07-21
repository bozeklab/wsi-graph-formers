"""
Lucas Sancéré 2025
"""

import sys
sys.path.append('../')  # Only for Remote use on Clusters

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
from torch_geometric.data import Data

from utils.graph_utils import _is_numeric, _flatten


def nx_to_pyg26_data(G: nx.Graph) -> Data:
    """
    Convert a NetworkX graph into a PyTorch-Geometric ``Data`` object.
    For PyG > 2.3 

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



def nx_to_pyg_data_manual(G: nx.Graph) -> Data:
    """
    Manual conversion of a NetworkX graph to a PyG Data object 
    (same behavior as from_networkx in PyG 2.6).
    For PyG < 2

    Parameters
    ----------
    G : networkx.Graph
        NetworkX graph with node features and 'cell_type' label.

    Returns
    -------
    Data
        PyG Data object with x, y, edge_index, and centroid.
    """
    # Feature keys (exclude 'cell_type')
    _, first_attr = next(iter(G.nodes(data=True)))
    feature_keys = sorted(
        k for k, v in first_attr.items()
        if k != "cell_type" and _is_numeric(v)
    )

    # Collect node features and labels
    xs, ys, centroids = [], [], []
    for _, attr in G.nodes(data=True):
        parts = [_flatten(attr.get(k, 0.0)) for k in feature_keys]
        xs.append(np.concatenate(parts).astype(np.float32))
        ys.append(int(attr.get("cell_type", -1)))
        centroids.append(attr.get("centroid", (0.0, 0.0)))  # fallback if missing

    x = torch.tensor(np.stack(xs), dtype=torch.float)
    y = torch.tensor(ys, dtype=torch.long)
    centroid = torch.tensor(centroids, dtype=torch.float)

    # Build edge index 
    edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()
    if edge_index.numel() == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    # Final PyG Data object 
    data = Data(x=x, y=y, edge_index=edge_index, centroid=centroid)
    return data




def save_skinwsi_graph(pyg_graph: Data, output_path: str):
    """
    Simple function to save with torch.save following the arguments of the skinwsi graphs Data
    """
    torch.save(
            {
            'x': pyg_graph.x,
            'y': pyg_graph.y,
            'edge_index': pyg_graph.edge_index,
            'centroid':pyg_graph.centroid,
            }, 
            output_path
    )




@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):

    # in case conversion_output_folder does not exiqsts 
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

            # we don't want the script to stop in case of a problem on one file 
            try:
                with open(pickle_path, "rb") as f:
                    G = pickle.load(f)
                pyg_graph = nx_to_pyg_data_manual(G)
                # Store attributes as dict to avoid version compatibility issues
                save_skinwsi_graph(pyg_graph, output_path)
                print(f"Converted graph saved to: {output_path}")

            # we do 2 exceptions: 
            # one specific to file  missing, not a pickle, or truncated 
            except (FileNotFoundError,
                    pickle.UnpicklingError,
                    EOFError) as e:

                print(f"Skipping {filename}: {e}")
                continue 

            # the other one for any other unexpected error:
            except Exception as e:

                print(f"Failed on {filename}: {e}")
                continue

    print("Conversion complete!")

if __name__ == "__main__":
    main()
