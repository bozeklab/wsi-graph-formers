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
from torch_geometric.utils import subgraph
import networkx as nx
from typing import Optional

from omegaconf import DictConfig
import hydra
from configs.schema import GraphConfig




def subgraph_filtering(data: Data, nodes_to_keep: torch.Tensor, filtering_step: int) -> Data:
    """
    Apply a subgraph operation to retain only a subset of nodes and
    update all corresponding node attributes accordingly.

    Parameters
    ----------
    data : torch_geometric.data.Data
        Input PyG graph containing edge_index, cell_type, and optionally x, y, centroid.
    nodes_to_keep : torch.Tensor
        1D tensor of node indices to retain in the filtered subgraph.
    filtering_step : int
        Optional flag for debugging or behavior control.

    Returns
    -------
    data : torch_geometric.data.Data
        Filtered graph containing only the selected nodes and updated attributes.
    """
    edge_index, _ = subgraph(
        nodes_to_keep,
        data['edge_index'],
        relabel_nodes=True,
        num_nodes=data['x'].shape[0]
    )
    data['edge_index'] = edge_index

    for key in ['x', 'y', 'cell_type', 'centroid']:
        if key in data:
            data[key] = data[key][nodes_to_keep]

    # for older pyg 
    # edge_index, _ = subgraph(
    #     nodes_to_keep,
    #     data.edge_index,
    #     relabel_nodes=True,
    #     num_nodes=data.num_nodes
    # )
    # data.edge_index = edge_index

    # # Filter node attributes
    # if hasattr(data, 'x'):
    #     data.x = data.x[nodes_to_keep]
    # if hasattr(data, 'y'):
    #     data.y = data.y[nodes_to_keep]

    return data



def simplify_graph(input_path: str, max_hops: Optional[int] = 3) -> Data:
    """
    Loads a PyG graph, removes background nodes (cell_type = 0 or label = -1, 
    i.e. unclassified nodes). If `max_hops` is specified, additionally removes
    nodes that are more than `max_hops` connections away from any normal or tumor 
    epithelial cell (cell_type = 5 and 6).

    Parameters
    ----------
    input_path : str
        Path to the saved PyG Data object (.pt file).
    max_hops : int or "Inf"
        Maximum allowed distance to retain nodes (in number of hops).
        If "Inf", only background removal is applied.

    Returns
    -------
    Data
        Filtered PyG graph.
    """
    # Load the full graph
    graph = torch.load(input_path)

    # STEP 1 — Remove all unclassified nodes (cell_type = 0 or label == -1)
    keep_mask = (graph['y'] != 0) & (graph['y'] != -1)
    classified_nodes = keep_mask.nonzero(as_tuple=True)[0]

    # Subgraph the PyG object to keep only valid nodes
    subgraph = subgraph_filtering(graph, classified_nodes, filtering_step=1)

    # If max_hops is None → return after background removal
    if str(max_hops) is "Inf":
        return subgraph

    else:
        # STEP 2 — Convert to NetworkX for shortest path analysis
        G_nx = nx.Graph()
        edge_list = subgraph['edge_index'].t().tolist()
        G_nx.add_edges_from(edge_list)

        # STEP 3 — Find all nodes within `max_hops` from any epithelial (5 or 6)
        target_types = {5, 6}
        anchor_nodes = [i for i, ct in enumerate(subgraph['y'].tolist()) if ct in target_types]

        nodes_to_keep = set()
        for node in anchor_nodes:
            if node in G_nx:
                neighbors = nx.single_source_shortest_path_length(G_nx, node, cutoff=max_hops)
                nodes_to_keep.update(neighbors.keys())

        # STEP 4 — Final subgraph with only close-enough nodes
        nodes_close2epithelial = torch.tensor(sorted(nodes_to_keep), dtype=torch.long)
        second_subgraph = subgraph_filtering(subgraph, nodes_close2epithelial, filtering_step=2)

        return second_subgraph






@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):

    # in case simplifiedgraph_folder does not exiqsts 
    if not os.path.exists(cfg.simplifiedgraph_folder):
        os.mkdir(cfg.simplifiedgraph_folder)

    
    suffix = '_simplified_' + str(cfg.max_hops) + '-hops-ngbr'
    outputext = '.pt'
    pygpaths = os.path.join(cfg.conversion_output_folder, '*.pt')
    pygpaths = glob.glob(pygpaths)


    for filepath in tqdm(pygpaths):
        if os.path.exists(filepath):
            pyg_path = filepath
            filename = str(os.path.splitext(os.path.split(filepath)[1])[0])
            output_path = cfg.simplifiedgraph_folder + filename + suffix + outputext

            graph = simplify_graph(pyg_path, cfg.max_hops)

            torch.save(
                    {
                    'x': graph['x'],
                    'y': graph['y'],
                    'edge_index': graph['edge_index'],
                    'centroid':graph['centroid'],
                    }, 
                    output_path
            )

            print(f"Converted graph saved to: {output_path}")

    print("Simplification complete!")


if __name__ == "__main__":
    main()