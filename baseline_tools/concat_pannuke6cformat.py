"""
Lucas Sancéré 2025
"""


#!/usr/bin/env python3
"""
Concatenate PanNuke-style tiles (PNG images + NPZ masks) into N folds of .npy arrays. Defaults to 3 folds and uses the **type_map** from your NPZs as the label.

Input directory structure (flat or nested works):
  <root>/.../<tile_name>.png
  <root>/.../<tile_name>.npz

Output directory structure:
  <out_root>/fold0/images.npy
  <out_root>/fold0/masks.npy
  <out_root>/fold1/images.npy
  <out_root>/fold1/masks.npy
  <out_root>/fold2/images.npy
  <out_root>/fold2/masks.npy

Notes
-----
- We auto-pair PNGs and NPZs by stem (filename without extension).
- The NPZs you generate contain `type_map` (uint8, classes 0..N with 0=background) and `inst_map` (int32 instance IDs). 
By default we read `type_map`. You can choose a specific key via `--mask-key`.
- All images and masks must share the same spatial shape (H, W). We'll infer channels from the first files.
- Saves as .npy using memmaps to be memory-safe for large datasets.


Example
-------
python pannuke_concat_to_folds.py \
    --input /path/to/tiles \
    --output /path/to/out_folds \
    --folds 3 \
    --seed 1337
"""


import json
import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from PIL import Image
from tqdm import tqdm

from omegaconf import DictConfig
import hydra
from hydra.utils import to_absolute_path

# ----------------------
# Helpers
# ----------------------


def _squeeze_hw(arr: np.ndarray) -> np.ndarray:
    """Return (H,W) for arrays that might be (H,W) or (H,W,1)."""
    if arr.ndim == 3 and arr.shape[2] == 1:
        return arr[..., 0]
    return arr


def count_cells_by_type(type_map: np.ndarray,
                        inst_map: np.ndarray,
                        classes: List[int] = None) -> Dict[int, int]:
    """
    Count distinct instance IDs (inst_map>0) for each semantic class in `classes`.
    Assumes your pipeline produces a uniform class per instance (as in your rasterization).
    Returns: {class_id: cell_count}
    """
    tmap = _squeeze_hw(np.asarray(type_map))
    inst = _squeeze_hw(np.asarray(inst_map))

    if tmap.shape != inst.shape:
        raise ValueError(f"type_map and inst_map shapes differ: {tmap.shape} vs {inst.shape}")

    counts: Dict[int, int] = {}
    labels = classes if classes is not None else [int(x) for x in np.unique(tmap) if x != 0]
    for lab in labels:
        if lab == 0:
            continue
        ids = np.unique(inst[(tmap == lab) & (inst > 0)])
        counts[int(lab)] = int(len(ids))
    return counts


def discover_pairs(root: Path) -> List[Tuple[Path, Path, str]]:
    """Return list of (png_path, npz_path, stem) for paired tiles.
    Pairs are matched by stem across .png and .npz.
    """
    pngs = {p.stem: p for p in root.rglob('*.png')}
    npzs = {p.stem: p for p in root.rglob('*.npz')}
    common = sorted(set(pngs.keys()) & set(npzs.keys()))
    pairs = [(pngs[s], npzs[s], s) for s in common]
    missing_png = sorted(set(npzs.keys()) - set(pngs.keys()))
    missing_npz = sorted(set(pngs.keys()) - set(npzs.keys()))
    if missing_png:
        print(f"[WARN] {len(missing_png)} masks without PNG: e.g., {missing_png[:3]}")
    if missing_npz:
        print(f"[WARN] {len(missing_npz)} PNGs without mask: e.g., {missing_npz[:3]}")
    if not pairs:
        raise RuntimeError("No PNG/NPZ pairs found.")
    return pairs


def load_image(path: Path) -> np.ndarray:
    img = Image.open(path).convert('RGB')
    arr = np.asarray(img)
    return arr


def extract_mask_from_npz(npz_path: Path, key_preference: str = 'type_map') -> np.ndarray:
    """Robustly extract a mask array from a NPZ file.
    Strategy: prefer common keys; otherwise pick the largest ndarray by .size.
    """
    data = np.load(npz_path, allow_pickle=True)
    candidate_keys = [k for k in data.files]
    key = None
    # 1) If caller asked for a specific key and it's present, use it
    if key_preference in data:
        key = key_preference
    else:
        # 2) Best-effort fallback: try common names
        preferred = ['type_map', 'inst_map', 'mask', 'masks', 'arr_0', 'label', 'labels']
        for k in preferred:
            if k in data:
                key = k
                break
    if key is None:
        # choose the largest array-like
        best_size = -1
        for k in candidate_keys:
            try:
                arr = data[k]
                if isinstance(arr, np.ndarray) and arr.size > best_size:
                    best_size = arr.size
                    key = k
            except Exception:
                continue
        if key is None:
            raise RuntimeError(f"No array found in NPZ: {npz_path}")
    arr = np.array(data[key])
    # Ensure mask is at least HxW (or HxWxC). If 2D int, keep; if 3D, keep as-is.
    if arr.ndim == 2:
        # semantic map single channel
        arr = arr[..., None]  # HxWx1
    elif arr.ndim == 3:
        pass
    else:
        # try to squeeze trailing singleton dims
        arr = np.squeeze(arr)
        if arr.ndim == 2:
            arr = arr[..., None]
        elif arr.ndim != 3:
            raise RuntimeError(f"Unsupported mask shape {arr.shape} in {npz_path}")
    # Normalize dtype to uint8 if possible to save space
    if np.issubdtype(arr.dtype, np.floating):
        # e.g., 0/1 floats -> uint8
        if np.nanmax(arr) <= 255 and np.nanmin(arr) >= 0:
            arr = arr.astype(np.uint8)
    elif arr.dtype == np.bool_:
        arr = arr.astype(np.uint8)
    return arr


def infer_shapes(sample_pair: Tuple[Path, Path, str], mask_key: str) -> Tuple[int, int, int, int]:
    img = load_image(sample_pair[0])
    msk = extract_mask_from_npz(sample_pair[1], key_preference=mask_key)
    H, W = img.shape[:2]
    if msk.shape[0] != H or msk.shape[1] != W:
        raise RuntimeError(
            f"Shape mismatch for {sample_pair[2]}: image {img.shape} vs mask {msk.shape}"
        )
    C_img = img.shape[2] if img.ndim == 3 else 1
    C_msk = msk.shape[2] if msk.ndim == 3 else 1
    return H, W, C_img, C_msk


def split_into_folds(stems: List[str], folds: int, seed: int) -> List[List[str]]:
    rng = np.random.default_rng(seed)
    order = np.array(stems)
    rng.shuffle(order)
    return [list(chunk) for chunk in np.array_split(order, folds)]


def write_fold(
    fold_idx: int,
    stems: List[str],
    pair_map: Dict[str, Tuple[Path, Path]],
    out_root: Path,
    H: int,
    W: int,
    C_img: int,
    C_msk: int,
    mask_key: str,
    pannuke: bool = False,           # kept for API compatibility; ignored when mask_key=='type_map'
    num_classes: int = 5,
    class_order: Optional[List[int]] = None,
) -> None:
    """
    Write one fold to disk:
      - images.npy  -> (N, H, W, C_img) uint8
      - masks.npy   -> (N, H, W, K) uint8  (K = num_classes when mask_key=='type_map',
                                            else K = C_msk)
      - stats.json  -> tiles_kept, cells_per_class, total_cells
    One-hot encoding is always used when mask_key=='type_map' (CellViT-friendly).
    Also collects cell counts per class using 'inst_map' if present.
    """
    fold_dir = out_root / f"fold{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    img_path = fold_dir / "images.npy"
    msk_path = fold_dir / "masks.npy"

    # Prepare class mapping/order
    if class_order is None:
        class_order = list(range(1, num_classes + 1))  # e.g., [1,2,3,4,5] or [1..6]
    label_to_ch: Dict[int, int] = {lab: ci for ci, lab in enumerate(class_order)}

    # One-hot is forced for type_map to satisfy prepare_pannuke.py
    save_onehot = (mask_key == "type_map")
    out_C_msk = num_classes if save_onehot else C_msk

    # Per-fold cell counts (for classes in class_order)
    cells_per_class: Dict[int, int] = {c: 0 for c in class_order}

    # Allocate memmaps
    N = len(stems)
    img_mm = np.lib.format.open_memmap(
        img_path, mode="w+", dtype=np.uint8, shape=(N, H, W, C_img)
    )
    msk_mm = np.lib.format.open_memmap(
        msk_path, mode="w+", dtype=np.uint8, shape=(N, H, W, out_C_msk)
    )

    # Iterate tiles
    for i, stem in enumerate(tqdm(stems, desc=f"fold{fold_idx}", unit="tile")):
        png_p, npz_p = pair_map[stem]

        # ---- load and validate image ----
        img = load_image(png_p)  # (H,W,3) RGB
        if img.shape[:2] != (H, W):
            raise RuntimeError(f"Image shape mismatch for {stem}: {img.shape} vs {(H, W)}")
        if img.ndim == 2:
            img = img[..., None]
        # enforce C_img
        if img.shape[2] != C_img:
            if img.shape[2] == 1 and C_img == 3:
                img = np.repeat(img, 3, axis=2)
            elif img.shape[2] != C_img:
                raise RuntimeError(f"Inconsistent image channels for {stem}: {img.shape[2]} vs {C_img}")
        img_mm[i] = img.astype(np.uint8)

        # ---- load label map (type_map or other) ----
        msk = extract_mask_from_npz(npz_p, key_preference=mask_key)  # (H,W) or (H,W,1)
        lab2d = _squeeze_hw(msk)
        if lab2d.shape != (H, W):
            raise RuntimeError(f"Mask shape mismatch for {stem}: {lab2d.shape} vs {(H, W)}")

        # ---- accumulate cell counts from inst_map when possible ----
        try:
            inst_map = extract_mask_from_npz(npz_p, key_preference="inst_map")
            inst2d = _squeeze_hw(inst_map)
            if inst2d.shape == (H, W):
                per_tile = count_cells_by_type(lab2d, inst2d, classes=class_order)
                for c, v in per_tile.items():
                    cells_per_class[c] = cells_per_class.get(c, 0) + int(v)
        except Exception:
            # if inst_map missing/malformed, skip counting for this tile
            pass

        # ---- write mask ----
        if save_onehot:
            # one-hot over 'num_classes' channels using class_order
            oh = np.zeros((H, W, num_classes), dtype=np.uint8)
            for lab_val, ch in label_to_ch.items():
                oh[..., ch] = (lab2d == lab_val).astype(np.uint8)
            msk_mm[i] = oh
        else:
            # generic path: preserve incoming channels (C_msk)
            if msk.ndim == 2:
                msk = msk[..., None]
            if msk.shape[2] != out_C_msk:
                if msk.shape[2] == 1 and out_C_msk > 1:
                    msk = np.repeat(msk, out_C_msk, axis=2)
                else:
                    raise RuntimeError(f"Inconsistent mask channels for {stem}: {msk.shape[2]} vs {out_C_msk}")
            if np.issubdtype(msk.dtype, np.floating):
                msk = np.nan_to_num(msk)
            msk_mm[i] = np.clip(msk, 0, 255).astype(np.uint8)

    # flush memmaps
    del img_mm
    del msk_mm

    # ---- write stats ----
    stats = {
        "tiles_kept": int(N),
        "cells_per_class": {str(c): int(cells_per_class.get(c, 0)) for c in sorted(cells_per_class)},
        "total_cells": int(sum(cells_per_class.values())),
    }
    with open(fold_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"[STATS] fold{fold_idx}: tiles={N} | cells: {stats['cells_per_class']}")



def compute_major_class_for_tiles(stems: List[str], pair_map: Dict[str, Tuple[Path, Path]], mask_key: str) -> Dict[str, int]:
    """Return a mapping stem -> majority class (ignoring 0) for simple stratification.
    Only meaningful for 'type_map'. If no foreground pixels, class 0 is used.
    """
    result: Dict[str, int] = {}
    for s in tqdm(stems, desc="majority-class", unit="tile"):
        npz_p = pair_map[s][1]
        m = extract_mask_from_npz(npz_p, key_preference=mask_key)
        if mask_key != 'type_map':
            result[s] = 0
            continue
        # m shape HxW or HxWx1
        if m.ndim == 3 and m.shape[2] == 1:
            m = m[..., 0]
        vals, counts = np.unique(m, return_counts=True)
        # ignore background 0 when deciding majority if possible
        fg = [(v, c) for v, c in zip(vals, counts) if v != 0]
        if fg:
            result[s] = int(max(fg, key=lambda vc: vc[1])[0])
        else:
            result[s] = 0
    return result


def stratified_split(stems: List[str], labels: Dict[str, int], folds: int, seed: int) -> List[List[str]]:
    rng = np.random.default_rng(seed)
    # group stems by label
    buckets: Dict[int, List[str]] = {}
    for s in stems:
        buckets.setdefault(int(labels.get(s, 0)), []).append(s)
    # shuffle each bucket, then distribute round-robin
    fold_lists = [[] for _ in range(folds)]
    for lab, items in buckets.items():
        items = items.copy()
        rng.shuffle(items)
        for i, s in enumerate(items):
            fold_lists[i % folds].append(s)
    # final shuffle within folds
    for lst in fold_lists:
        rng.shuffle(lst)
    return fold_lists





@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):

    # Resolve paths via Hydra's original CWD
    tiles_dir = Path(to_absolute_path(str(cfg.concat.tiles_dir)))
    out_root  = Path(to_absolute_path(str(cfg.concat.out_root)))

    folds     = int(cfg.concat.folds)
    seed      = int(cfg.concat.seed)
    stratify  = bool(cfg.concat.stratify)

    mask_key     = str(cfg.concat.mask_key)
    pannuke      = bool(cfg.concat.pannuke)
    num_classes  = int(cfg.concat.num_classes)
    class_map    = list(cfg.concat.class_map)
    start_at_one = bool(cfg.concat.fold_index_starts_at_one)

    out_root.mkdir(parents=True, exist_ok=True)



    # check if len(class_map) and num_classes are indeed matching
    if len(class_map) != num_classes:
        print(f"[WARN] class_map length {len(class_map)} != num_classes {num_classes}; "
              f"resetting to 1..{num_classes}")
        class_map = list(range(1, num_classes + 1))

    # Accept CSV string too
    if isinstance(class_map, str):
        try:
            class_map = [int(x.strip()) for x in class_map.split(',') if x.strip()]
        except Exception:
            class_map = [1,2,3,4,5]



    pairs = discover_pairs(tiles_dir)
    pair_map: Dict[str, Tuple[Path, Path]] = {s: (png, npz) for (png, npz, s) in pairs}
    stems = [s for (_, _, s) in pairs]

    H, W, C_img, C_msk = infer_shapes(pairs[0], mask_key)
    print(f"[INFO] Image shape: (H={H}, W={W}, C={C_img}); Mask channels: {C_msk} from '{mask_key}'")

    if stratify and mask_key == 'type_map':
        print('[INFO] Computing per-tile majority classes for stratification …')
        maj = compute_major_class_for_tiles(stems, pair_map, mask_key)
        fold_stems = stratified_split(stems, maj, folds, seed)
    else:
        fold_stems = split_into_folds(stems, folds, seed)

    # Optionally rename folds starting at 1
    def fold_name(i: int) -> str:
        return f"fold{i+1}" if start_at_one else f"fold{i}"

    # Write each fold
    # Note: write_fold internally creates out_root/fold{fi}
    for fi, stems_i in enumerate(fold_stems):
        print(f"[INFO] Writing {fold_name(fi)} with {len(stems_i)} tiles …")
        write_fold(
            fi, stems_i, pair_map, out_root, H, W, C_img, C_msk,
            mask_key, pannuke=pannuke, num_classes=num_classes,
            class_order=class_map,
        )
        print(f"[OK] {fold_name(fi)} saved to {out_root / fold_name(fi)}")

    print("[DONE] All folds written.")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.")
        sys.exit(130)
