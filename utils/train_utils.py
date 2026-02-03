import numpy as np
import re
import random
import torch




### create folds ###
def cv_train_test_indices(n, k_folds=5, testfold=0, seed=42):
    # used only if cv is True
    rng = np.random.RandomState(int(seed))
    idx = np.arange(n)
    rng.shuffle(idx)
    folds = np.array_split(idx, k_folds)

    if not (0 <= testfold < k_folds):
        raise ValueError(f"`fold` must be in [0, {k_folds-1}], got {testfold}")

    test_idx = folds[testfold]
    train_idx = np.concatenate([folds[i] for i in range(k_folds) if i != testfold])
    return train_idx, test_idx


### graph ID to split the graph correctly into folds 
def normalize_wsi(s: str) -> str:
    # collapse weird spacing so matching is stable
    return re.sub(r"\s+", " ", s).strip()


# depending on the naming of the graphs, get_graph_id or get_patchgraph_id 
def get_graph_id(item) -> str:
    graphname = item.name
    # path = _extract_path_from_item(item)

    base = graphname.split("_simplified_3-hops-ngbr_")[0]
    if base.startswith("subgraph_graph_r50_"):
        base = base[len("subgraph_graph_r50_"):]
    return normalize_wsi(base)

# depending on the naming of the graphs, get_graph_id or get_patchgraph_id 
def get_patchgraph_id(item) -> str:
    graphname = item.name
    # path = _extract_path_from_item(item)
    # patches odd counrts - tumor
    # patches even counts - healthy 

    base = graphname.split("_nosimplification_")[0]
    if base.startswith("graph_r50_mote_"):
        base = base[len("graph_r50_mote_"):]
    return normalize_wsi(base)



### function to check epithelial nodes 
def has_epithelial_nodes(g) -> bool:
    y = g.label
    if not torch.is_tensor(y):
        y = torch.as_tensor(y)

    return ((y == 4) | (y == 5)).any().item()

# def _extract_path_from_item(item):
#     # Case 1: already a path
#     if isinstance(item, (str, bytes, os.PathLike)):
#         return os.fspath(item)

#     # Case 2: PyG Data-like object
#     # Try common attributes
#     for attr in ["path", "filepath", "file_path", "filename", "fname", "name"]:
#         if hasattr(item, attr):
#             val = getattr(item, attr)
#             if isinstance(val, (str, bytes, os.PathLike)):
#                 return os.fspath(val)

#     # Case 3: stored in Data dict (PyG supports item['key'] sometimes)
#     for key in ["path", "filepath", "file_path", "filename", "fname", "name"]:
#         try:
#             val = item[key]
#             if isinstance(val, (str, bytes, os.PathLike)):
#                 return os.fspath(val)
#         except Exception:
#             pass

#     raise TypeError(
#         f"Cannot extract path from item of type {type(item)}. "
#         f"Available attrs: {dir(item)[:30]} ..."
#     )



# NOTE: for consistent data splits, see data_utils.rand_train_test_idx
def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
