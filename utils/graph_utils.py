"""
Lucas Sancéré 2025
"""

from __future__ import annotations
from typing import Any
import numpy as np
import torch
from torch import Tensor


def _is_numeric(value: Any) -> bool:  # noqa: D401  – keep the underscores
    """Return *True* iff *value* is numeric or a sequence of numeric scalars.

    Parameters
    ----------
    value : Any
        Candidate value to test. Allowed containers are ``list``, ``tuple``
        and :class:`numpy.ndarray`.

    Returns
    -------
    bool
        ``True`` when *value* is either a numeric scalar (``int``, ``float``,
        ``numpy.number``) or a homogeneous sequence of numeric scalars;
        ``False`` otherwise.
    """
    if np.isscalar(value):  # int, float, np.number …
        return True

    if isinstance(value, (list, tuple, np.ndarray)):
        # flatten first to support ragged n‑d arrays
        return all(np.isscalar(v) for v in np.ravel(value))

    return False



def _flatten(value: Any) -> np.ndarray:
    """Convert a numeric scalar/sequence into a 1‑D ``float32`` array.

    Non‑numeric objects yield an **empty** array so that callers can decide how
    to handle invalid inputs.

    Parameters
    ----------
    value : Any
        Numeric scalar or sequence (*list*, *tuple*, *ndarray*).

    Returns
    -------
    numpy.ndarray
        One‑dimensional ``float32`` representation of *value*.
    """
    if np.isscalar(value):
        return np.array([value], dtype=np.float32)

    if isinstance(value, (list, tuple, np.ndarray)):
        return np.asarray(value, dtype=np.float32).ravel()

    # not numeric –> empty vector so feature concatenation still works
    return np.empty(0, dtype=np.float32)



def fit_stats_pyg(train_list, cont_idx, centroid_idx=None, device='cpu'):
    """
    Compute mean and standard deviation of continuous node features
    across a list of training graphs for later z-score normalization.

    Parameters
    ----------
    train_list : list of torch_geometric.data.Data
        List of training graphs, each containing node features `x`
        and optionally a `centroid` tensor.
    cont_idx : list of int
        Indices of continuous features inside `data.x` (excluding
        cell_type and centroid).
    centroid_idx : list of int or None, optional
        Indices of centroid coordinates inside `data.x`. If None,
        the centroid is taken from `data.centroid` (Nx2).
    device : str, optional
        Device on which to perform the computation. Default is 'cpu'.

    Returns
    -------
    stats : dict
        Dictionary with keys:
        - "mu" : torch.Tensor
            Mean of concatenated continuous features across all nodes.
        - "sd" : torch.Tensor
            Standard deviation of concatenated continuous features.
    """
    xs = []
    for data in train_list:
        x = data.x.to(device)
        if centroid_idx is None:
            c = data.centroid.to(device).float()
        else:
            c = x[:, centroid_idx].float()
        cont = x[:, cont_idx].float()
        block = torch.cat([c, cont], dim=1)
        xs.append(block)
    X = torch.cat(xs, dim=0)
    mu = X.mean(dim=0)
    sd = X.std(dim=0)
    sd = torch.where(sd == 0, torch.ones_like(sd), sd)
    return {"mu": mu.cpu(), "sd": sd.cpu()}



def transform_pyg(
    data,
    stats,
    *,
    cont_idx,
    cell_type_idx=None,
    num_cell_classes=6,
    gamma=3.0,
    centroid_idx=None,
    normalize_centroid=None
):
    """
    Transform a PyG graph by z-scoring continuous features and
    up-weighting the cell_type feature via scaled one-hot encoding.

    Parameters
    ----------
    data : torch_geometric.data.Data
        Input PyG graph with node attributes `x`, and either a
        separate `centroid` tensor (Nx2) or centroid columns inside `x`.
    stats : dict
        Dictionary containing "mu" and "sd" tensors from fit_stats_pyg.
    cont_idx : list of int
        Indices of continuous features inside `data.x` (excluding
        cell_type and centroid).
    cell_type_idx : int or None, optional
        Index of cell_type column inside `data.x` if present.
        If None, the function expects `data.cell_type` as an attribute.
    num_cell_classes : int, optional
        Number of unique cell types. Default is 6.
    gamma : float, optional
        Scaling factor applied to the one-hot encoding of cell_type.
        Controls the relative weight of the cell_type feature.
    centroid_idx : list of int or None, optional
        Indices of centroid columns inside `data.x`. If None, centroid
        is taken from `data.centroid`.
    normalize_centroid : tuple(int, int) or None, optional
        If provided, (width, height) are used to normalize centroid
        coordinates to [0,1] before z-scoring. Default is None.

    Returns
    -------
    data : torch_geometric.data.Data
        Graph with updated `data.x` containing the transformed feature
        vector [z-scored continuous || γ·onehot(cell_type)].
        Raw attributes (e.g., `data.x_raw`, `data.centroid`) are kept
        for reproducibility. Metadata such as feature names, stats, and
        gamma are attached to the graph.
    """
    device = data.x.device
    mu = stats["mu"].to(device)
    sd = stats["sd"].to(device)

    # Build continuous block
    if centroid_idx is None:
        c = data.centroid.float()
        if normalize_centroid is not None:
            W, H = normalize_centroid
            c = torch.stack([c[:, 0] / float(W), c[:, 1] / float(H)], dim=1)
    else:
        c = data.x[:, centroid_idx].float()
        if normalize_centroid is not None:
            W, H = normalize_centroid
            c = torch.stack([c[:, 0] / float(W), c[:, 1] / float(H)], dim=1)

    cont = data.x[:, cont_idx].float()
    cont_block = torch.cat([c, cont], dim=1)

    # z-score
    z = (cont_block - mu) / sd

    # Cell type one-hot (scaled)
    if cell_type_idx is not None:
        ct = data.x[:, cell_type_idx].view(-1)
    elif hasattr(data, "cell_type"):
        ct = data.cell_type.view(-1)
    else:
        raise ValueError("cell_type not found: provide cell_type_idx or data.cell_type")

    oh = torch.zeros(ct.numel(), num_cell_classes, device=ct.device)
    ct0 = (ct.long() - 1).clamp(0, num_cell_classes - 1)
    oh.scatter_(1, ct0.unsqueeze(1), 1.0)
    oh = oh * float(gamma)

    # Concatenate final features
    new_x = torch.cat([z, oh], dim=1)

    # Preserve raw x
    data.x_raw = data.x
    data.x = new_x
    data.feature_names = [
        "centroid_x","centroid_y",
        *[f"cont_{i}" for i in cont_idx],
        *[f"cell_{k}" for k in range(1, num_cell_classes+1)],
    ]
    data.transform_mu = mu.detach().cpu()
    data.transform_sd = sd.detach().cpu()
    data.gamma_cell_type = float(gamma)
    data.transform_version = "ct_onehot_gamma_v1"
    return data



# Typical usage

# # Suppose your current data has:
# # - data.x with columns: [cell_type, area, perimeter, eccentricity, solidity, major_axis_length, minor_axis_length, extent, tex_* ...]
# # - data.centroid as (N,2)
# # Then:
# cell_type_idx = 0
# cont_idx = list(range(1, data.x.size(1)))  # all but cell_type
# centroid_idx = None  # because centroid is separate

# # Fit stats on TRAIN graphs
# stats = fit_stats_pyg(train_list, cont_idx=cont_idx, centroid_idx=centroid_idx)

# # Transform any split
# gamma = 3.0
# for data in [train_data, val_data, test_data]:
#     transform_pyg(
#         data, stats,
#         cont_idx=cont_idx,
#         cell_type_idx=cell_type_idx,      # or None if you have data.cell_type
#         num_cell_classes=6,
#         gamma=gamma,
#         centroid_idx=None,                # using data.centroid
#         normalize_centroid=None           # or (W, H) if you want [0,1] coords firs