"""
Lucas Sancéré 2025
"""

from __future__ import annotations
from typing import Any
import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import Subset


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



def fit_zscore_stats_pyg(
    train_list, 
    cont_idx, 
    centroid_idx=None, 
    normalize_centroid=None,
    device='cpu'):
    """
    Compute mean and standard deviation of continuous node features
    (centroid + specified continuous columns) across training graphs
    for later z-score normalization.

    Parameters
    ----------
    train_list : list of torch_geometric.data.Data
        List of training graphs, each containing node features `x`
        and either a separate `centroid` tensor (Nx2) or centroid
        columns inside `x`.
    cont_idx : list of int
        Indices of continuous features inside `data.x` (excluding
        any categorical columns).
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
            Mean of the concatenated [centroid || continuous] block across all nodes.
        - "sd" : torch.Tensor
            Standard deviation of the same block (zeros are replaced with ones).
    """
    xs = []
    for data in train_list:
        x = data.x.to(device)
        if centroid_idx is None:
            c = data.centroid.to(device).float()
        else:
            c = x[:, centroid_idx].float()
        if normalize_centroid is not None:
            W, H = normalize_centroid
            c = torch.stack([c[:, 0] / float(W), c[:, 1] / float(H)], dim=1)
        cont = x[:, cont_idx].float()
        xs.append(torch.cat([c, cont], dim=1))
    X = torch.cat(xs, dim=0)
    mu = X.mean(dim=0)
    sd = X.std(dim=0)
    sd = torch.where(sd == 0, torch.ones_like(sd), sd)
    return {"mu": mu.cpu(), "sd": sd.cpu()}



def normalize_zscore_pyg(
    data,
    stats,
    *,
    cont_idx,
    centroid_idx=None,
    normalize_centroid=None
):
    """
    Transform a PyG graph by z-scoring continuous features only.
    The output feature matrix is [z-scored centroid || z-scored continuous].

    Parameters
    ----------
    data : torch_geometric.data.Data
        Input PyG graph with node attributes `x`, and either a
        separate `centroid` tensor (Nx2) or centroid columns inside `x`.
    stats : dict
        Dictionary containing "mu" and "sd" tensors from
        `fit_zscore_stats_pyg`. These are used for z-score scaling.
    cont_idx : list of int
        Indices of continuous features inside `data.x` (excluding
        categorical columns).
    centroid_idx : list of int or None, optional
        Indices of centroid columns inside `data.x`. If None, centroid
        is taken from `data.centroid`.
    normalize_centroid : tuple(int, int) or None, optional
        If provided, (width, height) are used to normalize centroid
        coordinates to [0,1] before z-scoring. Default is None.

    Returns
    -------
    data : torch_geometric.data.Data
        Graph with updated `data.x` containing only the transformed
        feature vector [z-scored centroid || z-scored continuous].
        Raw attributes are preserved in `data.x_raw`.
        Metadata (feature names, stats, version) are attached for
        reproducibility.
    """
    device = data.x.device
    mu = stats["mu"].to(device)
    sd = stats["sd"].to(device)

    # Build continuous block [centroid || cont]
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

    # Preserve raw x, replace with z-scored block only
    data.x_raw = data.x
    data.x = z
    data.feature_names = [
        "centroid_x","centroid_y",
        *[f"cont_{i}" for i in cont_idx],
    ]
    data.transform_mu = mu.detach().cpu()
    data.transform_sd = sd.detach().cpu()
    data.transform_version = "zscore_v1"
    return data




def append_celltype_onehot_pyg(
    data,
    *,
    num_cell_classes: int = 6,
    gamma: float = 1.0,
    attach: bool = True,
    position: str = "append",
    unknown_label: int = -1,   # <-- 0 is now a valid class; -1 denotes "unknown"
    label_base: int = 0,       # 0 for labels in [0..K-1], 1 for labels in [1..K]
):
    """
    Append a scaled one-hot encoding of `data.label` to existing node features.

    Label conventions
    -----------------
    - `unknown_label` is treated as "no label" → all-zeros row. Default: -1.
    - If `label_base=0`, valid class ids are integers in [0..num_cell_classes-1].
    - If `label_base=1`, valid class ids are integers in [1..num_cell_classes].
      (Internally shifted to 0..K-1 for one-hot.)

    Parameters
    ----------
    data : torch_geometric.data.Data
        Input PyG graph with `x` (optional), and `label` (N,) for cell types.
        Unknown labels are allowed. `num_nodes`, `x`, or `centroid` must
        exist to determine the number of nodes.
    num_cell_classes : int, optional
        Number of unique cell types/classes (K). Default is 6.
    gamma : float, optional
        Scaling factor applied to the one-hot encoding. Default is 6.0.
    attach : bool, optional
        If True, modifies the graph in-place:
          - Preserves old features in `data.x_raw` (if not already set).
          - Replaces `data.x` with [existing features || γ·onehot(label)].
          - Updates `data.feature_names`, `data.gamma_cell_type`,
            and `data.transform_version`.
        If False, only returns the concatenated tensor. Default is True.
    position : {"append", "prepend"}, optional
        Whether to append the one-hot block after existing features ("append")
        or prepend it before ("prepend"). Default is "append".
    unknown_label : int, optional
        Integer in `data.label` that denotes "no label". Default is -1.
    label_base : int, optional
        0 if class ids in `data.label` are 0..K-1; 1 if they are 1..K. Default 0.

    Returns
    -------
    new_x : torch.Tensor
        Feature matrix with shape (N, D + num_cell_classes), where D is the
        original feature dimension. If `attach=True`, also modifies `data`
        in-place as described above.
    """
    # ---- Determine device and N ------------------------------------------------
    if hasattr(data, "x") and data.x is not None:
        device = data.x.device
        N = data.x.size(0)
    else:
        device = torch.device("cpu")
        if getattr(data, "num_nodes", None) is not None:
            N = int(data.num_nodes)
        elif getattr(data, "centroid", None) is not None:
            N = data.centroid.size(0)
        else:
            raise ValueError("Cannot infer number of nodes (need data.x, data.num_nodes, or data.centroid).")

    # ---- Build one-hot -------------------------------------
    # Start with an all-zero (N, K) matrix.
    K = int(num_cell_classes)
    oh = torch.zeros(N, K, device=device)

    # If labels exist, normalize dtype, handle NaNs, and mark valid rows.
    if hasattr(data, "label") and data.label is not None:
        lbl = data.label.view(-1)

        # Convert to long; if float, protect against NaNs first.
        if torch.is_floating_point(lbl):
            lbl = lbl.clone()
            nan_mask = torch.isnan(lbl)
            lbl[nan_mask] = unknown_label
            lbl = lbl.to(torch.long)
        else:
            lbl = lbl.to(torch.long)

        # Make a mask for rows that have a usable class id
        # (i.e., not unknown and within the expected range).
        if label_base == 0:
            valid_range = (lbl >= 0) & (lbl < K)
            class_idx = lbl.clone()               # already 0..K-1
        elif label_base == 1:
            valid_range = (lbl >= 1) & (lbl <= K)
            class_idx = lbl.clone() - 1           # shift to 0..K-1
        else:
            raise ValueError("label_base must be 0 or 1.")

        valid = (lbl != unknown_label) & valid_range

        # Fill only valid rows using F.one_hot for clarity.
        if valid.any():
            oh_valid = F.one_hot(class_idx[valid], num_classes=K).to(oh.dtype)
            oh[valid] = oh_valid

    # Scale the one-hot block.
    oh = oh * float(gamma)

    # ---- Concatenate with existing features -----------------------------------
    if hasattr(data, "x") and data.x is not None:
        if data.x.size(0) != N:
            raise ValueError(f"data.x has {data.x.size(0)} rows but inferred N={N}.")
        new_x = torch.cat([oh, data.x], dim=1) if position == "prepend" else torch.cat([data.x, oh], dim=1)
    else:
        new_x = oh

    # ---- Attach or return ------------------------------------------------------
    if attach:
        if not hasattr(data, "x_raw"):
            data.x_raw = getattr(data, "x", None)

        data.x = new_x

        # Feature names: keep existing if provided; otherwise fabricate.
        prev_names = getattr(data, "feature_names", None)
        if prev_names is None and getattr(data, "x_raw", None) is not None:
            prev_dim = data.x_raw.size(1)
            prev_names = [f"feat_{i}" for i in range(prev_dim)]
        elif prev_names is None and new_x is oh:
            prev_names = []

        # Name the one-hot channels; match label_base for readability.
        if label_base == 0:
            oh_names = [f"cell_{k}" for k in range(0, K)]
        else:
            oh_names = [f"cell_{k}" for k in range(1, K + 1)]

        data.feature_names = ([*oh_names, *(prev_names or [])]
                              if position == "prepend"
                              else [*(prev_names or []), *oh_names])

        data.gamma_cell_type = float(gamma)
        data.transform_version = "append_label_onehot_v2"
        data.unknown_label_value = int(unknown_label)
        data.label_base = int(label_base)
        return data.x

    return new_x





def normalize_encode_celltype_pyg(
    data,
    stats,
    *,
    cont_idx,
    num_cell_classes=6,
    gamma=6.0,
    centroid_idx=None,
    normalize_centroid=None,
    unknown_label: int = -1,
    label_base: int = 0,
):
    """
    Transform a PyG graph by z-scoring continuous features and appending a
    scaled one-hot encoding derived from `data.label`.

    Label conventions
    -----------------
    - `unknown_label` is treated as "no label" → all-zeros row. Default: -1.
    - If `label_base=0`, valid class ids are integers in [0..num_cell_classes-1].
    - If `label_base=1`, valid class ids are integers in [1..num_cell_classes]
      (internally shifted to 0..K-1 for one-hot).

    Output
    ------
    x = [z-scored centroid || z-scored continuous || γ·onehot(label)]

    Parameters
    ----------
    data : torch_geometric.data.Data
        Input PyG graph with node attributes `x`, and either a separate
        `centroid` tensor (Nx2) or centroid columns inside `x`. Must have
        `data.label` (N,) for cell types; unknowns are allowed.
    stats : dict
        Dictionary containing "mu" and "sd" tensors from `fit_zscore_stats_pyg`.
    cont_idx : list of int
        Indices of continuous features inside `data.x` (excluding categorical).
    num_cell_classes : int, optional
        Number of unique cell types/classes. Default is 6.
    gamma : float, optional
        Scaling factor applied to the one-hot encoding. Default is 6.0.
    centroid_idx : list of int or None, optional
        Indices of centroid columns inside `data.x`. If None, centroid
        is taken from `data.centroid`.
    normalize_centroid : tuple(int, int) or None, optional
        If provided, (width, height) are used to normalize centroid
        coordinates to [0,1] before z-scoring.
    unknown_label : int, optional
        Integer in `data.label` that denotes "no label". Default is -1.
    label_base : int, optional
        0 if class ids in `data.label` are 0..K-1; 1 if they are 1..K. Default 0.

    Returns
    -------
    data : torch_geometric.data.Data
        Graph with updated `data.x`:
        [z-scored centroid || z-scored continuous || γ·onehot(label)].
        Raw attributes (e.g., `data.x_raw`) are preserved. Metadata such as
        feature names, stats, gamma, and version are attached.
    """
    device = data.x.device
    mu = stats["mu"].to(device)
    sd = stats["sd"].to(device)

    # ----- Build continuous block [centroid || cont] -----
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

    # ----- z-score -----
    z = (cont_block - mu) / sd

    # ----- Cell type one-hot from data.label (readable) -----
    N = z.size(0)
    K = int(num_cell_classes)
    oh = torch.zeros(N, K, device=device)

    if hasattr(data, "label") and data.label is not None:
        lbl = data.label.view(-1)

        # dtype & NaN handling
        if torch.is_floating_point(lbl):
            lbl = lbl.clone()
            nan_mask = torch.isnan(lbl)
            lbl[nan_mask] = unknown_label
            lbl = lbl.to(torch.long)
        else:
            lbl = lbl.to(torch.long)

        # Map to internal 0..K-1 indices and mark valid rows
        if label_base == 0:
            class_idx = lbl.clone()                       # already 0..K-1 if valid
            valid_range = (class_idx >= 0) & (class_idx < K)
        elif label_base == 1:
            class_idx = lbl.clone() - 1                   # shift to 0..K-1
            valid_range = (class_idx >= 0) & (class_idx < K)
        else:
            raise ValueError("label_base must be 0 or 1.")

        valid = (lbl != unknown_label) & valid_range

        if valid.any():
            oh_valid = F.one_hot(class_idx[valid], num_classes=K).to(oh.dtype)
            oh[valid] = oh_valid

    # Scale one-hot
    oh = oh * float(gamma)

    # ----- Concatenate final features -----
    new_x = torch.cat([z, oh], dim=1)

    # ----- Preserve raw x & attach metadata -----
    data.x_raw = data.x
    data.x = new_x

    if label_base == 0:
        cell_names = [f"cell_{k}" for k in range(0, K)]
    else:
        cell_names = [f"cell_{k}" for k in range(1, K + 1)]

    data.feature_names = [
        "centroid_x", "centroid_y",
        *[f"cont_{i}" for i in cont_idx],
        *cell_names,
    ]
    data.transform_mu = mu.detach().cpu()
    data.transform_sd = sd.detach().cpu()
    data.gamma_cell_type = float(gamma)
    data.unknown_label_value = int(unknown_label)
    data.label_base = int(label_base)
    data.transform_version = "zscore_plus_label_onehot_v2"
    return data



# def mask_celltype_onehot_cols_(data, classes, *, label_base=0):
#     fn = getattr(data, "feature_names", None)
#     if fn is None:
#         return False
#     name_to_col = {n: i for i, n in enumerate(fn)}
#     cols = []
#     for c in classes:
#         key = f"cell_{c}" if label_base in (0, 1) else None
#         if key in name_to_col:
#             cols.append(name_to_col[key])
#     if not cols:
#         return False
#     data.x[:, cols] = 0.0
#     # optional breadcrumbs:
#     data.celltype_masked_classes = tuple(sorted(set(classes)))
#     return True



# def mask_on_graph_list(graphs, classes, label_base=0, verbose=False):
#     seen = 0
#     masked = 0
#     for d in graphs:
#         seen += 1
#         masked += 1 if mask_celltype_onehot_cols_(d, classes, label_base=label_base) else 0
#     if verbose:
#         print(f"Masked {masked}/{seen} graphs (label_base={label_base}, classes={classes}).")




def mask_celltype_onehot_cols(data, classes, *, label_base=0):
    """
    In-place: set the one-hot columns for given classes to 0 in data.x.
    Requires data.feature_names from append_celltype_onehot_pyg. No-op if absent.
    """
    fn = getattr(data, "feature_names", None)
    if fn is None:
        return
    name_to_col = {n: i for i, n in enumerate(fn)}
    cols = [name_to_col.get(f"cell_{c}") for c in classes]
    cols = [c for c in cols if c is not None]
    if not cols:
        return
    data.x[:, cols] = 0.0



# def mask_celltype_onehot(target, classes_to_mask=(4, 5), *, label_base=0):
#     """
#     In-place: zero the cell-type one-hot columns for the given classes
#     across all graphs in `target`.

#     Parameters
#     ----------
#     target : Data | list[Data] | Dataset | Subset | DataLoader | list of these
#         Object(s) containing PyG Data graphs.
#     classes_to_mask : tuple[int], optional
#         Class IDs whose one-hot columns should be zeroed. Use the same ID
#         convention as `label_base`. Default: (4, 5).
#     label_base : int, optional
#         0 if classes are named "cell_0..cell_{K-1}", 1 if "cell_1..cell_{K}".
#         Default: 0.

#     Notes
#     -----
#     - Requires `data.feature_names` to locate the one-hot columns (from
#       `append_celltype_onehot_pyg`). If not present, the graph is skipped.
#     - This zeros the selected columns for **all rows** (nodes) of each graph.
#     """
#     def _mask_one_graph(data):
#         if not hasattr(data, "feature_names") or data.feature_names is None:
#             return
#         name_to_col = {n: i for i, n in enumerate(data.feature_names)}
#         wanted = [f"cell_{c}" for c in classes_to_mask] if label_base in (0, 1) else []
#         cols = [name_to_col[n] for n in wanted if n in name_to_col]
#         if cols:
#             data.x[:, cols] = 0.0  # in-place

#     def _apply(obj):
#         # Unwrap DataLoader -> dataset
#         if hasattr(obj, "dataset") and not hasattr(obj, "x"):
#             return _apply(obj.dataset)

#         # Unwrap Subset recursively
#         if isinstance(obj, Subset):
#             return _apply(obj.dataset)

#         # List/Tuple of things
#         if isinstance(obj, (list, tuple)):
#             for item in obj:
#                 _apply(item)
#             return

#         # Dataset-like
#         if hasattr(obj, "__len__") and hasattr(obj, "__getitem__") and not hasattr(obj, "x"):
#             # Try to modify returned graphs; assign back when possible
#             for i in range(len(obj)):
#                 g = obj[i]
#                 _mask_one_graph(g)
#                 try:
#                     obj[i] = g
#                 except Exception:
#                     pass
#             return

#         # Single graph
#         if hasattr(obj, "x"):  # PyG Data
#             _mask_one_graph(obj)

#     _apply(target)


# def install_mask_celltype_onehot(targets, classes_to_mask=(4, 5), *, label_base=0):
#     """
#     Attach a transform that zeros the specified one-hot columns in `data.x`
#     for every graph yielded by the given loaders/datasets.

#     Call once before training:
#         install_mask_celltype_onehot([train_loader, val_loader, test_loader],
#                                      classes_to_mask=(4,5),
#                                      label_base=cfg.label_base)
#     """
#     def mask_once(data):
#         # Only touch the one-hot block; require feature_names from append_celltype_onehot_pyg
#         fn = getattr(data, "feature_names", None)
#         if fn is None:
#             return data
#         name_to_col = {n: i for i, n in enumerate(fn)}
#         wanted = [f"cell_{c}" for c in classes_to_mask] if label_base in (0, 1) else []
#         cols = [name_to_col[n] for n in wanted if n in name_to_col]
#         if cols:
#             data.x[:, cols] = 0.0  # in-place, nothing else touched
#         return data

#     def add_transform(ds):
#         # unwrap Subset to the real dataset carrying the transform
#         while isinstance(ds, Subset):
#             ds = ds.dataset
#         base = getattr(ds, "transform", None)
#         if base is None:
#             ds.transform = mask_once
#         else:
#             # compose: base first, then mask
#             def chained(d, b=base):
#                 return mask_once(b(d))
#             ds.transform = chained

#     def handle(obj):
#         if obj is None:
#             return
#         # If it's a DataLoader, recurse into its dataset
#         if hasattr(obj, "dataset") and not hasattr(obj, "x"):
#             return handle(obj.dataset)
#         # If it's a list/tuple of things, apply to each
#         if isinstance(obj, (list, tuple)):
#             for it in obj:
#                 handle(it)
#             return
#         # Dataset-like (has __len__/__getitem__)
#         if hasattr(obj, "__len__") and hasattr(obj, "__getitem__") and not hasattr(obj, "x"):
#             add_transform(obj)
#             return
#         # Single Data graph
#         if hasattr(obj, "x"):
#             mask_once(obj)

#     handle(targets)


