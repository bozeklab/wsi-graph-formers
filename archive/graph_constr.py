"""
Lucas Sancéré 2025
"""



import json
import numpy as np
from skimage.draw import polygon
from skimage import measure
from scipy.spatial import cKDTree
import networkx as nx
from tqdm import tqdm

# Load JSON
with open('path/to/your_file.json', 'r') as f:
    data = json.load(f)

cells = data.get("cells", data)  # support both top-level or under "cells"

ids = []
centroids = []
features = []
cell_types = []

print("Parsing cell features...")
for cell_id, cell_data in tqdm(cells.items()):
    try:
        cell_id = int(cell_id)
    except:
        pass
    ids.append(cell_id)
    cell_types.append(cell_data.get("type", -1))
    centroid = cell_data["centroid"]
    centroids.append(centroid)

    contour = np.array(cell_data["contour"])
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack([contour, contour[0]])
    x, y = contour[:, 0], contour[:, 1]
    min_x, max_x = int(np.floor(x.min())), int(np.ceil(x.max()))
    min_y, max_y = int(np.floor(y.min())), int(np.ceil(y.max()))
    mask = np.zeros((max_y - min_y + 1, max_x - min_x + 1), dtype=np.uint8)
    rr, cc = polygon(y - min_y, x - min_x)
    mask[rr, cc] = 1

    props = measure.regionprops(mask)
    if props:
        p = props[0]
        feats = [
            p.area,
            p.perimeter,
            p.eccentricity,
            p.solidity,
            p.major_axis_length,
            p.minor_axis_length,
            p.extent
        ]
    else:
        feats = [0.0] * 7

    features.append(feats)

centroids = np.array(centroids)
features = np.array(features)

# Build graph using radius-based spatial indexing
print("Building spatial index...")
tree = cKDTree(centroids)
radius = 50  # pixels; adjust based on image scale

print(f"Finding neighbors within radius {radius}...")
pairs = tree.query_pairs(r=radius)

# Build graph
print("Constructing graph...")
G = nx.Graph()
for i, node_id in enumerate(ids):
    G.add_node(node_id,
               cell_type=cell_types[i],
               area=features[i, 0],
               perimeter=features[i, 1],
               eccentricity=features[i, 2],
               solidity=features[i, 3],
               major_axis_length=features[i, 4],
               minor_axis_length=features[i, 5],
               extent=features[i, 6])

for i, j in tqdm(pairs):
    G.add_edge(ids[i], ids[j])  # no distance attribute

print(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# Optional: Convert to PyG format
try:
    from torch_geometric.utils import from_networkx
    data = from_networkx(G, 
        node_attrs=["area", "perimeter", "eccentricity", "solidity", "major_axis_length", "minor_axis_length", "extent", "cell_type"]
    )
except ImportError:
    data = None

# data is now ready for PyTorch Geometric use, or G for NetworkX analysis
