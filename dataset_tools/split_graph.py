"""
Lucas Sancéré 2025
"""

import sys
sys.path.append('../')  # Only for Remote use on Cluste

import os
from tqdm import tqdm
import glob
import torch
from torch_geometric.data import Data
from torch_geometric.utils import subgraph as node_subgraph
import networkx as nx
from typing import Dict

from omegaconf import DictConfig
import hydra
from configs.schema import GraphConfig

from models.SGFormer.largewsi.dataset import NCDataset





def extract_cluster_subgraph_dict(inputgraph, node_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Induced subgraph on nodes where `node_mask` is True, returned as a dict with:
    {'x', 'y', 'edge_index', 'centroid'}.

    Parameters
    ----------
    inputgraph : object
        Holds:
          - edge_index : LongTensor[2, E]
          - node_feat  : Tensor[N, F]
          - centroids  : Tensor[N, d]
          - num_nodes  : int
          - label      : Tensor[...] (node-level or graph-level)
          - edge_feat  : Optional[Tensor[E, ...]] (ignored for output)
    node_mask : torch.Tensor
        Bool mask of shape [N] selecting nodes to keep.

    Returns
    -------
    dict
        {
          'x'         : Tensor[n', F],
          'y'         : Tensor[n'] (node-level) or original label (graph-level),
          'edge_index': LongTensor[2, e'],
          'centroid'  : Tensor[n', d],
        }
    """
    if node_mask.dtype != torch.bool:
        node_mask = node_mask.bool()

    N = int(inputgraph.num_nodes)
    node_idx = node_mask.nonzero(as_tuple=False).view(-1)      # [n']

    # Original graph
    ei = inputgraph.edge_index                                  # [2, E]
    ea = getattr(inputgraph, "edge_feat", None)                 # optional, not returned

    # Induced subgraph with relabeling (PyG 1.7-safe)
    new_ei, _ = node_subgraph(
        subset=node_idx,
        edge_index=ei,
        edge_attr=ea,           # passed only so PyG filters consistently; we don't return it
        relabel_nodes=True,
        num_nodes=N,
    )

    # x / centroid
    x_out = inputgraph.node_feat[node_idx] if getattr(inputgraph, "node_feat", None) is not None else None
    c_out = inputgraph.centroids[node_idx] if getattr(inputgraph, "centroids", None) is not None else None

    # y: node-level vs graph-level
    y_in = getattr(inputgraph, "label", None)
    if y_in is not None and torch.is_tensor(y_in) and y_in.dim() > 0 and y_in.size(0) == N:
        y_out = y_in[node_idx]          # node-level labels
    else:
        y_out = y_in                    # graph-level label or None

    return {
        'x': x_out,
        'y': y_out,
        'edge_index': new_ei,
        'centroid': c_out,
    }




def torch_kmeans_subgraphs(
    inputgraph,
    K: int,
    num_iters: int = 20,
    tol: float = 1e-4
    ) -> Dict[str, torch.Tensor]:
    """
    Partition nodes into K spatial clusters via k-means on `inputgraph.centroids`
    and return the induced subgraphs (edges preserved within clusters).

    Parameters
    ----------
    inputgraph : object
        An object with attributes:
            - edge_index : LongTensor[2, E]
            - node_feat  : Tensor[N, F]
            - edge_feat  : Optional[Tensor[E, ...]]
            - num_nodes  : int
            - centroids  : Tensor[N, d]
            - label      : Optional[Tensor or any]
    K : int
        Number of spatial clusters (subgraphs) to produce. Must be in [1, num_nodes].
    num_iters : int
        Maximum number of k-means updates. Defaults to 20.
    tol : float
        Convergence threshold for centroid movement (Euclidean norm). Defaults to 1e-4.

    Returns
    -------
    List[torch_geometric.data.Data]
        A list of length K. Each entry is a PyG `Data` object containing only nodes
        assigned to that spatial cluster, with `edge_index` relabelled and attributes sliced.
    """
    assert hasattr(inputgraph, "centroids") and inputgraph.centroids is not None, \
        "`inputgraph.centroids` (node positions) is required."
    assert 1 <= K <= int(inputgraph.num_nodes), "K must be between 1 and the number of nodes."

    centroids = inputgraph.centroids  # [N, d]
    N = centroids.size(0)

    # 1) Initialize cluster centers by sampling K nodes
    init_idx = torch.randperm(N)[:K]
    centers = centroids[init_idx].clone()  # [K, d]

    for _ in range(num_iters):
        # 2) Assign each node to nearest center (Euclidean)
        dists = torch.cdist(centroids, centers, p=2.0)  # [N, K]
        labels = dists.argmin(dim=1)                    # [N]

        # 3) Update centers as mean of assigned nodes (keep if empty)
        new_centers = []
        for k in range(K):
            mk = labels == k
            new_centers.append(centroids[mk].mean(dim=0) if mk.any() else centers[k])
        new_centers = torch.stack(new_centers, dim=0)

        # 4) Convergence check
        if torch.norm(new_centers - centers) < tol:
            centers = new_centers
            break
        centers = new_centers

    # 5) Build induced subgraphs for each cluster (edges preserved inside) (dicts)
    return [extract_cluster_subgraph_dict(inputgraph, labels == k) for k in range(K)]



def save_subgraphs(
    subgraphs: Dict[str, torch.Tensor],
    output_dir: str,
    inputname: str,
    prefix: str = "subgraph"
    ) -> None:
    """
    Save a list of PyG subgraphs as separate .pt files.

    Parameters
    ----------
    subgraphs : List[torch_geometric.data.Data]
        List of PyG Data objects (e.g., output from `torch_kmeans_subgraphs`).
    output_dir : str
        Path to the folder where subgraphs should be saved.
    prefix : str, optional
        Filename prefix for the saved subgraphs. Defaults to "subgraph".

    Returns
    -------
    None
        Files are written to `output_dir` in the form `<prefix>_k.pt` for k=0..len-1.
    """
    os.makedirs(output_dir, exist_ok=True)

    for i, sg in enumerate(subgraphs):
        idx = i+1
        fname = os.path.join(output_dir, f"{prefix}_{inputname}_{idx}.pt")
        torch.save(sg, fname)





@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):

    # in case split_output_folder does not exiqsts 
    if not os.path.exists(cfg.split_output_folder):
        os.mkdir(cfg.split_output_folder)

    # choose if we split from the simplified graphs or from the original pyg ones
    if cfg.split_from_simplified:
        split_input_folder = cfg.simplifiedgraph_folder 
    else:
        split_input_folder = cfg.conversion_output_folder

    pygpaths = os.path.join(split_input_folder, '*.pt')
    pygpaths = glob.glob(pygpaths)

    print("Subgraphs generation...")

    for filepath in tqdm(pygpaths):
        if os.path.exists(filepath):
            pyg_path = filepath
            filename = str(os.path.splitext(os.path.split(filepath)[1])[0])

            # Load the graph 
            graph_dict = torch.load(pyg_path)

            # For this step it is just a dict and not pyg data format
            # convert it to pyg Data
            inputgraph = NCDataset('onegraphskinwsi')

            edge_index = graph_dict['edge_index']
            node_feat = graph_dict['x']
            label = graph_dict['y']
            num_nodes = node_feat.shape[0]
            # here we will need to save the centroids as well 
            #in order to create the subgraphs 
            centroids = graph_dict['centroid']

            # inputgraph.graph = {
            #     'edge_index': edge_index,
            #     'node_feat': node_feat,
            #     'edge_feat': None,
            #     'num_nodes': num_nodes,
            # }
            inputgraph.edge_index = edge_index
            inputgraph.node_feat = node_feat
            inputgraph.edge_feat = None
            inputgraph.num_nodes = num_nodes
            inputgraph.centroids = centroids
            inputgraph.label = label 

            # Create subgraphs
            subgraphs = torch_kmeans_subgraphs(inputgraph, K=cfg.nbr_subgraphs)

            # Save them
            save_subgraphs(subgraphs, 
                output_dir=cfg.split_output_folder,
                inputname=filename, 
                prefix="subgraph"
                )

    print("All subgraphs generated!")


if __name__ == "__main__":
    main()
