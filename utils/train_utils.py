import numpy as np
import pandas as pd
import re
import random
import torch
from collections import defaultdict
from pathlib import Path



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


def cv_subgraphs_skinwsi_patient_split(
    graph_list,
    patient_csv,
    *,
    wsigraph: bool,
    k_folds: int,
    testfold: int,
    seed: int,
    normalize_wsi_fn,
    get_graph_id_fn,
    get_patchgraph_id_fn,
    has_epithelial_nodes_fn,
    verbose: bool = True,
):
    """
    Cross-validation split for subgraphs-skinwsi:
      - keep ONLY Tumor == Yes WSIs (from metadata)
      - keep ONLY subgraphs having at least one epithelial node
      - group subgraphs by WSI, then by patient
      - split by patient into train/test folds (no leakage)

    Returns
    -------
    train_data, test_data, val_data (empty list), info dict
    """

    # ---- load metadata, keep tumor WSIs only
    meta = pd.read_csv(patient_csv, sep=";")
    meta["Tile_graphs"] = meta["Tile_graphs"].map(normalize_wsi_fn)
    meta["Tumor"] = meta["Tumor"].astype(str).str.strip()

    tumor_meta = meta[meta["Tumor"].str.lower().eq("yes")].copy()
    wsi_to_patient = dict(zip(tumor_meta["Tile_graphs"], tumor_meta["Patient_ID"]))
    tumor_wsis = set(wsi_to_patient.keys())

    # ---- filter graphs to tumor WSIs only
    if wsigraph:
        graph_list_tumor = [g for g in graph_list if get_graph_id_fn(g) in tumor_wsis]
    else:
        graph_list_tumor = [g for g in graph_list if get_patchgraph_id_fn(g) in tumor_wsis]

    # ---- keep subgraphs with epithelial nodes
    graph_list_epi = [g for g in graph_list_tumor if has_epithelial_nodes_fn(g)]

    # ---- group by WSI
    groups = defaultdict(list)  # wsi -> [subgraphs...]
    for g in graph_list_epi:
        gid = get_graph_id_fn(g) if wsigraph else get_patchgraph_id_fn(g)
        groups[gid].append(g)

    # ---- group WSIs by patient
    patients = defaultdict(list)  # pid -> [wsi...]
    for wsi in groups.keys():
        pid = wsi_to_patient[wsi]
        patients[pid].append(wsi)

    patient_ids = list(patients.keys())
    n_patients = len(patient_ids)
    if n_patients < k_folds:
        raise ValueError(
            f"Not enough patients for {k_folds}-fold CV: got {n_patients} patients."
        )

    # ---- CV indices on patients (deterministic via seed)
    train_p_idx, test_p_idx = cv_train_test_indices(
        n_patients, k_folds=k_folds, testfold=testfold, seed=seed
    )
    train_pids = [patient_ids[i] for i in train_p_idx]
    test_pids  = [patient_ids[i] for i in test_p_idx]
    val_pids   = []  # keep empty to match your current CV logic

    # ---- expand back to subgraphs
    train_data, test_data, val_data = [], [], []

    for pid in train_pids:
        for wsi in patients[pid]:
            train_data.extend(groups[wsi])

    for pid in test_pids:
        for wsi in patients[pid]:
            test_data.extend(groups[wsi])

    # ---- leakage checks (WSI + patient disjoint)
    if wsigraph:
        train_wsis = {get_graph_id_fn(x) for x in train_data}
        test_wsis  = {get_graph_id_fn(x) for x in test_data}
    else:
        train_wsis = {get_patchgraph_id_fn(x) for x in train_data}
        test_wsis  = {get_patchgraph_id_fn(x) for x in test_data}
    assert train_wsis.isdisjoint(test_wsis), "WSI leakage between train and test."

    train_pids_check = {wsi_to_patient[wsi] for wsi in train_wsis}
    test_pids_check  = {wsi_to_patient[wsi] for wsi in test_wsis}
    assert train_pids_check.isdisjoint(test_pids_check), "Patient leakage between train and test."

    info = {
        "n_graphs_in": len(graph_list),
        "n_tumor_graphs": len(graph_list_tumor),
        "n_epi_graphs": len(graph_list_epi),
        "n_wsis_used": len(groups),
        "n_patients_used": len(patient_ids),
        "train_patients": len(train_pids),
        "test_patients": len(test_pids),
        "train_subgraphs": len(train_data),
        "test_subgraphs": len(test_data),
    }

    if verbose:
        # show which WSIs were discarded due to Tumor != Yes
        all_wsis_meta = set(meta["Tile_graphs"])
        kept_wsis_meta = set(tumor_meta["Tile_graphs"])
        discarded_wsis_by_tumor = sorted(all_wsis_meta - kept_wsis_meta)

        print(f"\nDiscarded WSIs (Tumor != Yes): {len(discarded_wsis_by_tumor)}")
        for w in discarded_wsis_by_tumor[:50]:
            print("  ", w)

        print("\nTumor subgraphs total:", len(graph_list_tumor))
        print("Tumor subgraphs with epithelial:", len(graph_list_epi))

        print(f"\nPatients total: {info['n_patients_used']}")
        print(f"Train patients: {info['train_patients']}  Test patients: {info['test_patients']}")
        print(f"Train subgraphs: {info['train_subgraphs']}  Test subgraphs: {info['test_subgraphs']}")

        # --------------------------------
        # PRINT FILE LIST PER FOLD
        # --------------------------------

        # extract WSI ids from actual train/test graphs
        if wsigraph:
            train_wsis = sorted({get_graph_id_fn(g) for g in train_data})
            test_wsis  = sorted({get_graph_id_fn(g) for g in test_data})
        else:
            train_wsis = sorted({get_patchgraph_id_fn(g) for g in train_data})
            test_wsis  = sorted({get_patchgraph_id_fn(g) for g in test_data})

        print("\n========== TRAIN WSIs ==========")
        print(f"Count: {len(train_wsis)}")
        for w in train_wsis:
            print(f"{w}    (Patient: {wsi_to_patient[w]})")

        print("\n========== TEST WSIs ==========")
        print(f"Count: {len(test_wsis)}")
        for w in test_wsis:
            print(f"{w}    (Patient: {wsi_to_patient[w]})")

        # also show patient list explicitly
        print("\n========== TRAIN PATIENTS ==========")
        for pid in sorted(train_pids):
            print(pid)

        print("\n========== TEST PATIENTS ==========")
        for pid in sorted(test_pids):
            print(pid)

        print("\n====================================\n")


    return train_data, test_data, val_data, info


### graph ID to split the graph correctly into folds 
def normalize_wsi(s: str) -> str:
    # collapse weird spacing so matching is stable
    return re.sub(r"\s+", " ", s).strip()


# depending on the naming of the graphs, get_graph_id or get_patchgraph_id 
def get_graph_id(item) -> str:

    # "wsigraph_93_4.pt" -> "wsigraph_93"
    stem = Path(item.name).stem          # drops ".pt"
    return stem.rsplit("_", 1)[0]        # drops the trailing "_4"

    # OLD reference:
    # graphname = item.name
    # # path = _extract_path_from_item(item)

    # base = graphname.split("_simplified_3-hops-ngbr_")[0]
    # if base.startswith("subgraph_graph_r50_"):
    #     base = base[len("subgraph_graph_r50_"):]
    # return normalize_wsi(base)
    


# depending on the naming of the graphs, get_graph_id or get_patchgraph_id 
def get_patchgraph_id(item) -> str:
    # "tilegraph_93_4.pt" -> "tilegraph_93"
    stem = Path(item.name).stem          # drops ".pt"
    return stem.rsplit("_", 1)[0]        # drops the trailing "_4"


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
