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
- The NPZs you generate contain `type_map` (uint8, classes 0..5 with 0=background) and `inst_map` (int32 instance IDs). By default we read `type_map`. You can choose a specific key via `--mask-key`.
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
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image
from tqdm import tqdm

from omegaconf import DictConfig
import hydra
from hydra.utils import to_absolute_path

# ----------------------
# Helpers
# ----------------------

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
    pannuke: bool = False,
    num_classes: int = 5,
    class_order: List[int] | None = None,
):
    fold_dir = out_root / f"fold{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    img_path = fold_dir / 'images.npy'
    msk_path = fold_dir / 'masks.npy'  # stores chosen mask_key array per tile

    # Create memmaps
    img_mm = np.lib.format.open_memmap(
        img_path, mode='w+', dtype=np.uint8, shape=(len(stems), H, W, C_img)
    )
    # masks: for PanNuke we want one-hot channels=num_classes; else keep C_msk
    out_C_msk = num_classes if pannuke and mask_key == 'type_map' else C_msk
    msk_mm = np.lib.format.open_memmap(
        msk_path, mode='w+', dtype=np.uint8, shape=(len(stems), H, W, out_C_msk)
    )

    # Prepare class order mapping
    if pannuke and mask_key == 'type_map':
        if class_order is None:
            # default assumes labels 1..num_classes map to channels 0..num_classes-1
            class_order = list(range(1, num_classes + 1))
        # map label -> channel index
        label_to_ch = {lab: ci for ci, lab in enumerate(class_order)}

    for i, stem in enumerate(tqdm(stems, desc=f"fold{fold_idx}", unit="tile")):
        png_p, npz_p = pair_map[stem]
        img = load_image(png_p)
        msk = extract_mask_from_npz(npz_p, key_preference=mask_key)
        # Basic validations
        if img.shape[:2] != (H, W):
            raise RuntimeError(f"Image shape mismatch for {stem}: {img.shape} vs {(H, W)}")
        if msk.shape[0] != H or msk.shape[1] != W:
            raise RuntimeError(f"Mask shape mismatch for {stem}: {msk.shape} vs {(H, W)}")
        if img.ndim == 2:
            img = img[..., None]
        if msk.ndim == 2:
            msk = msk[..., None]
        if img.shape[2] != C_img:
            raise RuntimeError(f"Inconsistent image channels for {stem}: {img.shape}")

        img_mm[i] = img.astype(np.uint8)

        # --- PanNuke one-hot conversion ---
        if pannuke and mask_key == 'type_map':
            # ensure we have 2D label map
            lab = msk[..., 0] if msk.ndim == 3 else msk
            # Build one-hot (H,W,num_classes)
            oh = np.zeros((H, W, num_classes), dtype=np.uint8)
            # assign each specified label to its channel
            for lab_val, ch in label_to_ch.items():
                oh[..., ch] = (lab == lab_val).astype(np.uint8)
            msk_mm[i] = oh
        else:
            # Generic path: clip to uint8, broadcast if needed
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

    # Ensure data is flushed
    del img_mm
    del msk_mm
    fold_dir = out_root / f"fold{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    img_path = fold_dir / 'images.npy'
    msk_path = fold_dir / 'masks.npy'  # stores chosen mask_key array per tile

    # Create memmaps
    img_mm = np.lib.format.open_memmap(
        img_path, mode='w+', dtype=np.uint8, shape=(len(stems), H, W, C_img)
    )
    # masks: pick a compact dtype (uint8), adjust if needed later
    msk_mm = np.lib.format.open_memmap(
        msk_path, mode='w+', dtype=np.uint8, shape=(len(stems), H, W, C_msk)
    )

    for i, stem in enumerate(tqdm(stems, desc=f"fold{fold_idx}", unit="tile")):
        png_p, npz_p = pair_map[stem]
        img = load_image(png_p)
        msk = extract_mask_from_npz(npz_p, key_preference=mask_key)
        # Basic validations
        if img.shape[:2] != (H, W):
            raise RuntimeError(f"Image shape mismatch for {stem}: {img.shape} vs {(H, W)}")
        if msk.shape[0] != H or msk.shape[1] != W:
            raise RuntimeError(f"Mask shape mismatch for {stem}: {msk.shape} vs {(H, W)}")
        if img.ndim == 2:
            img = img[..., None]
        if msk.ndim == 2:
            msk = msk[..., None]
        if img.shape[2] != C_img:
            raise RuntimeError(f"Inconsistent image channels for {stem}: {img.shape}")
        if msk.shape[2] != C_msk:
            # If C differs but one is 1 and the other is N, try to broadcast single-channel
            if msk.shape[2] == 1 and C_msk > 1:
                msk = np.repeat(msk, C_msk, axis=2)
            else:
                raise RuntimeError(f"Inconsistent mask channels for {stem}: {msk.shape[2]} vs {C_msk}")

        img_mm[i] = img.astype(np.uint8)
        # try to fit into uint8; if out-of-range, clip
        if np.issubdtype(msk.dtype, np.floating):
            msk = np.nan_to_num(msk)
        msk_mm[i] = np.clip(msk, 0, 255).astype(np.uint8)

    # Ensure data is flushed
    del img_mm
    del msk_mm


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
    """Hydra entry point. Reads config keys if present; falls back to PanNuke defaults.

    Expected config keys (put them anywhere in your cfg; shown here under `concat`):

    concat:
      tiles_dir: /abs/or/rel/path/to/tiles   # required
      out_root:  ./out_folds                 # default
      folds:     3                           # default PanNuke-style
      seed:      1337                        # default
      stratify:  false                       # default
      mask_key:  type_map                    # default
      pannuke:   true                        # default (one-hot 5 channels)
      num_classes: 5                         # default PanNuke
      class_map: [1,2,3,4,5]                 # default class order
      fold_index_starts_at_one: false        # default False → fold0, fold1, fold2
    """
    section = getattr(cfg, 'concat', cfg)

    # Resolve paths via Hydra's original CWD
    tiles_dir = Path(to_absolute_path(str(getattr(section, 'tiles_dir', '.'))))
    out_root  = Path(to_absolute_path(str(getattr(section, 'out_root', './out_folds'))))

    folds     = int(getattr(section, 'folds', 3))
    seed      = int(getattr(section, 'seed', 1337))
    stratify  = bool(getattr(section, 'stratify', False))

    mask_key  = str(getattr(section, 'mask_key', 'type_map'))
    pannuke   = bool(getattr(section, 'pannuke', True))
    num_classes = int(getattr(section, 'num_classes', 5))
    class_map = getattr(section, 'class_map', [1,2,3,4,5])
    # Accept CSV string too
    if isinstance(class_map, str):
        try:
            class_map = [int(x.strip()) for x in class_map.split(',') if x.strip()]
        except Exception:
            class_map = [1,2,3,4,5]
    start_at_one = bool(getattr(section, 'fold_index_starts_at_one', False))

    out_root.mkdir(parents=True, exist_ok=True)

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
