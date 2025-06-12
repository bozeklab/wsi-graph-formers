"""
Lucas Sancéré 2025
"""


import os
import glob
import json
import numpy as np
import networkx as nx
from skimage.draw import polygon
from skimage import measure
from scipy.spatial import cKDTree
from tqdm import tqdm
import os
from omegaconf import DictConfig
import hydra
from hydra.core.config_store import ConfigStore
from configs.schema import GraphConfig
import pickle
import time 
from contextlib import contextmanager

cs = ConfigStore.instance()
cs.store(name="graph_config", node=GraphConfig)



@contextmanager
def timing_block(timing=False, label="Elapsed Time"):
    if timing:
        start_time = time.perf_counter()
        yield
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        minutes = int(elapsed // 60)
        seconds = elapsed % 60
        print(f"{label}: {minutes} min {seconds:.2f} sec")
    else:
        yield



def extract_morph_features(contour):
    """
    Extract morphological features from a cell contour.

    Parameters
    ----------
    contour : np.ndarray
        Array of (x, y) coordinates representing the cell contour.

    Returns
    -------
    list
        List of morphological features: area, perimeter, eccentricity, solidity,
        major_axis_length, minor_axis_length, extent.
    """
    x, y = contour[:, 0], contour[:, 1]
    min_x, max_x = int(np.floor(x.min())), int(np.ceil(x.max()))
    min_y, max_y = int(np.floor(y.min())), int(np.ceil(y.max()))
    mask = np.zeros((max_y - min_y + 1, max_x - min_x + 1), dtype=np.uint8)
    rr, cc = polygon(y - min_y, x - min_x)
    mask[rr, cc] = 1

    props = measure.regionprops(mask)
    if props:
        p = props[0]
        return [
            p.area, p.perimeter, p.eccentricity, p.solidity,
            p.major_axis_length, p.minor_axis_length, p.extent
        ]
    else:
        return [0.0] * 7




def build_graph_from_json(json_path, radius=50, timing=False):
    """
    Build a cell graph from JSON input using radius-based neighbor pruning.

    Parameters
    ----------
    json_path : str
        Path to the JSON input file.
    radius : float
        Distance threshold (in pixels) for connecting neighboring cells.

    Returns
    -------
    G : networkx.Graph
        Graph with nodes containing morphology and type information.
    """
    with timing_block(timing, "Reading JSON"):
        with open(json_path, 'r') as f:
            data = json.load(f)

    cells = data.get("cells", data)
    ids, centroids, features, cell_types = [], [], [], []


    for cell_id, cell_data in tqdm(cells.items(), desc="Parsing cells"):
        try:
            cell_id = int(cell_id)
        except:
            pass
        ids.append(cell_id)
        cell_types.append(cell_data.get("type", -1))
        centroids.append(cell_data["centroid"])

        contour = np.array(cell_data["contour"])
        if not np.array_equal(contour[0], contour[-1]):
            contour = np.vstack([contour, contour[0]])

        feats = extract_morph_features(contour)
        features.append(feats)

    centroids = np.array(centroids)
    features = np.array(features)
     
    #for reproducibility - but can take some time to run
    ids = np.argsort(ids)

    tree = cKDTree(centroids)
    pairs = tree.query_pairs(r=radius)

    G = nx.Graph()
    for i, node_id in enumerate(ids):
        G.add_node(node_id,
                   cell_type=cell_types[i],
                   centroid=tuple(centroids[i]),  
                   area=features[i, 0],
                   perimeter=features[i, 1],
                   eccentricity=features[i, 2],
                   solidity=features[i, 3],
                   major_axis_length=features[i, 4],
                   minor_axis_length=features[i, 5],
                   extent=features[i, 6])


    for i, j in tqdm(pairs, desc="Adding edges"):
        G.add_edge(ids[i], ids[j])

    return G



def save_graph(graph, output_path, timing=False):
    """
    Save the constructed graph to a file.

    Parameters
    ----------
    graph : networkx.Graph
        The graph object to save.
    output_path : str
        Path to the output .gpickle file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with timing_block(timing, "Saving graph"):
        with open(output_path, "wb") as f:
            pickle.dump(graph, f)
    print(f"Graph saved to: {output_path}")




@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):

    outputext = '.pickle'
    jsonpaths = os.path.join(cfg.json_folder, '*.json')
    jsonpaths = glob.glob(jsonpaths)
    for filepath in tqdm(jsonpaths):
        if os.path.exists(filepath):
            json_path = filepath
            filename = str(os.path.splitext(os.path.split(filepath)[1])[0])
            output_path = cfg.output_folder + cfg.output_corename + filename + outputext  

            G = build_graph_from_json(json_path, radius=cfg.radius, timing=cfg.timing)
            save_graph(G, output_path,  timing=cfg.timing)

if __name__ == "__main__":
    main()
