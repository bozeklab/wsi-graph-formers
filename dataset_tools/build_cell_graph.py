"""
Lucas Sancéré 2025
"""

import sys
sys.path.append('../')  # Only for Remote use on Cluste

import os
import glob
import json
import numpy as np
import networkx as nx
from skimage.draw import polygon
from skimage import measure
from skimage.color import rgb2gray
from skimage.feature import graycomatrix, graycoprops
from skimage.util import img_as_ubyte
from scipy.spatial import cKDTree
from tqdm import tqdm
import os
from omegaconf import DictConfig
import hydra
from configs.schema import GraphConfig
import pickle
import time 
from contextlib import contextmanager

import openslide 



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




def read_bbox_patch(slide, bbox, level=0):
    """
    Read an RGB patch from the WSI at given bbox.
    
    Parameters
    ----------
    slide : openslide.OpenSlide
        Opened WSI handle.
    bbox : list
        [[xmin, ymin], [xmax, ymax]] in level-0 coordinates.
    level : int
        Pyramid level (0 = highest resolution).
    
    Returns
    -------
    np.ndarray or None
        Patch as HxWx3 uint8 array, or None if bbox is invalid.
    """
    xmin, ymin = map(int, bbox[0])
    xmax, ymax = map(int, bbox[1])

    # ensure positive width/height
    w, h = xmax - xmin, ymax - ymin
    if w <= 0 or h <= 0:
        return None

    # clip to slide dimensions
    W0, H0 = slide.level_dimensions[0]
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(W0, xmax)
    ymax = min(H0, ymax)
    w, h = xmax - xmin, ymax - ymin
    if w <= 0 or h <= 0:
        return None

    # read region
    region = slide.read_region((xmin, ymin), level, (w, h))  # PIL RGBA
    if region.mode != "RGB":
        region = region.convert("RGB")
    return np.asarray(region, dtype=np.uint8)  # HxWx3




def _entropy_from_hist(hist):
    p = hist.astype(np.float64)
    s = p.sum()
    if s <= 0:
        return 0.0
    p /= s
    nz = p[p > 0]
    return float(-(nz * np.log2(nz)).sum())



def extract_texture_features(patch_rgb,
                             glcm_distances=(1, 2, 4),
                             glcm_angles=(0, np.pi/4, np.pi/2, 3*np.pi/4),
                             gl_levels=32,
                             entropy_bins=256,
                             eps=1e-12
                            ):
    """
    Extract texture features from an image patch.

    Parameters
    ----------
    patch_rgb : np.ndarray
        Image patch, HxW, HxWx3 or HxWx4.
    glcm_distances : tuple[int]
        Distances for GLCM.
    glcm_angles : tuple[float]
        Angles for GLCM.
    gl_levels : int
        Number of gray levels for GLCM.
    entropy_bins : int
        Number of bins for entropy histogram.
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    list
        [roughness, contrast, dissimilarity, homogeneity,
         entropy, angular_second_moment, dispersion]
    """
    # --- grayscale [0,1]
    arr = np.asarray(patch_rgb)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        gray = rgb2gray(arr)
    elif arr.ndim == 2:
        gray = arr.astype(np.float32)
        if gray.max() > 1.0 + 1e-6:
            gray /= 255.0
        gray = np.clip(gray, 0.0, 1.0)
    else:
        raise ValueError("patch must be HxW or HxWxC with C>=3")

    mean_int = float(gray.mean())
    std_int  = float(gray.std())
    roughness = std_int
    dispersion = float(std_int / (mean_int + eps))

    # entropy
    gray_u8 = img_as_ubyte(gray)
    hist, _ = np.histogram(gray_u8, bins=entropy_bins, range=(0, 256))
    entropy = _entropy_from_hist(hist)

    # quantize for GLCM
    if gl_levels < 256:
        q = np.floor(gray_u8.astype(np.float32) * (gl_levels / 256.0)).astype(np.uint8)
        q[q >= gl_levels] = gl_levels - 1
        levels = gl_levels
    else:
        q = gray_u8
        levels = 256

    H, W = q.shape
    if H < 2 or W < 2:
        return [roughness, np.nan, np.nan, np.nan, entropy, np.nan, dispersion]

    glcm = graycomatrix(
        q,
        distances=glcm_distances,
        angles=glcm_angles,
        levels=levels,
        symmetric=True,
        normed=True
    )

    contrast      = float(graycoprops(glcm, "contrast").mean())
    dissimilarity = float(graycoprops(glcm, "dissimilarity").mean())
    homogeneity   = float(graycoprops(glcm, "homogeneity").mean())
    asm           = float(graycoprops(glcm, "ASM").mean())

    return [
        roughness,
        contrast,
        dissimilarity,
        homogeneity,
        entropy,
        asm,          # Angular Second Moment
        dispersion
    ]




def build_graph_from_json(json_path,
                          wsi_path,
                          with_texture=True,
                          timing=False,
                          radius=50,
                          level=0,
                          glcm_distances=(1, 2, 4),
                          glcm_angles=(0, np.pi/4, np.pi/2, 3*np.pi/4),
                          gl_levels=32
                          ):
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

    slide = openslide.OpenSlide(wsi_path)

    ids, centroids, cell_types = [], [], []
    morph_list, texture_list = [], []

    for cell_id, cell_data in tqdm(cells.items(), desc="Parsing cells"):
        try:
            cell_id = int(cell_id)
        except Exception:
            pass
        ids.append(cell_id)
        cell_types.append(cell_data.get("type", -1))
        centroids.append(cell_data["centroid"])

        contour = np.array(cell_data["contour"])
        if not np.array_equal(contour[0], contour[-1]):
            contour = np.vstack([contour, contour[0]])
        morph_list.append(extract_morph_features(contour))

        if with_texture:
            bbox = cell_data["bbox"]
            patch = read_bbox_patch(slide, bbox, level=level)
            if patch is None:
                texture_list.append([np.nan]*7)
            else:
                texture_list.append(
                    extract_texture_features(
                        patch,
                        glcm_distances=glcm_distances,
                        glcm_angles=glcm_angles,
                        gl_levels=gl_levels,
                    )
                )

    slide.close()

    # arrays + sort
    ids = np.array(ids)
    centroids = np.array(centroids, dtype=float)
    morph_arr = np.array(morph_list, dtype=float)
    cell_types = np.array(cell_types)
    if with_texture:
        texture_arr = np.array(texture_list, dtype=float)

    order = np.argsort(ids)
    ids = ids[order]
    centroids = centroids[order]
    morph_arr = morph_arr[order]
    cell_types = cell_types[order]
    if with_texture:
        texture_arr = texture_arr[order]

    # neighbors
    tree = cKDTree(centroids)
    pairs = tree.query_pairs(r=radius)


    N = ids.shape[0]

    # Build graph with consecutive labels 0..N-1, but keep original cell id
    G = nx.Graph()
    for i in range(N):
        attrs = dict(
            orig_id=int(ids[i]),                      # keep original label for traceability
            cell_type=int(cell_types[i]),
            centroid=tuple(centroids[i]),
            area=float(morph_arr[i, 0]),
            perimeter=float(morph_arr[i, 1]),
            eccentricity=float(morph_arr[i, 2]),
            solidity=float(morph_arr[i, 3]),
            major_axis_length=float(morph_arr[i, 4]),
            minor_axis_length=float(morph_arr[i, 5]),
            extent=float(morph_arr[i, 6]),
        )
        if with_texture:
            attrs.update(
                tex_roughness=float(texture_arr[i, 0]),
                tex_contrast=float(texture_arr[i, 1]),
                tex_dissimilarity=float(texture_arr[i, 2]),
                tex_homogeneity=float(texture_arr[i, 3]),
                tex_entropy=float(texture_arr[i, 4]),
                tex_angular_second_moment=float(texture_arr[i, 5]),
                tex_dispersion=float(texture_arr[i, 6]),
            )
        G.add_node(i, **attrs)  # <- consecutive node ids

    # KDTree pairs were computed on the sorted arrays, so (i, j) are 0..N-1 already
    for i, j in tqdm(pairs, desc="Adding edges"):
        if i != j:
            G.add_edge(int(i), int(j))

    return G



def save_pickle_graph(graph, output_path, timing=False):
    """
    Save the constructed graph to a file.

    Parameters
    ----------
    graph : networkx.Graph
        The graph object to save.
    output_path : str
        Path to the output .pickle file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with timing_block(timing, "Saving graph"):
        with open(output_path, "wb") as f:
            pickle.dump(graph, f)
    print(f"Graph saved to: {output_path}")




@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):

    outputext = '.pickle'
    jsonpaths = os.path.join(cfg.json_folder, '*.json')
    jsonpaths = glob.glob(jsonpaths)

    wsi_input_folder = cfg.wsi_input_folder

    for filepath in tqdm(jsonpaths):
        if os.path.exists(filepath):
            json_path = filepath
            filename = str(os.path.splitext(os.path.split(filepath)[1])[0])
            output_path = cfg.pickle_output_folder + cfg.pickle_output_corename + filename + outputext  

            wsi_path = wsi_input_folder + filename + '.ndpi'

            G = build_graph_from_json(
                json_path,
                wsi_path,
                with_texture=cfg.with_texture,
                radius=cfg.radius, 
                timing=cfg.timing
                )
            save_pickle_graph(G, output_path, timing=cfg.timing)

if __name__ == "__main__":
    main()
