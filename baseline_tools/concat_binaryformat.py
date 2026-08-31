"""
Lucas Sancéré 2025
"""

"""
Concatenate tiles for binary-vs-binary training with ternary labels:
  - 0 = background/other
  - 1 = (original) class 5
  - 2 = (original) class 6

Inclusion rule
--------------
A tile is INCLUDED only if >= ratio_threshold of NON-BACKGROUND pixels (label != 0)
are of class 5 or 6. Default ratio_threshold = 0.90.

For included tiles, all labels not in {5,6} are converted to 0, and {5,6} map to {1,2}.

Input directory (flat or nested):
  <root>/.../<tile>.png
  <root>/.../<tile>.npz   (expects key 'type_map' by default)

Outputs per fold:
  <out_root>/foldK/images.npy         (uint8, shape (N,H,W,3))
  <out_root>/foldK/masks.npy          (uint8, shape (N,H,W,1); values in {0,1,2})
  <out_root>/foldK/stats.json         (cell counts for original labels 5 and 6)

Example:
  python concat_binaryformat.py \
    concat_binary.tiles_dir=/path/to/tiles \
    concat_binary.out_root=./out_binary_56 \
    concat_binary.folds=3 \
    concat_binary.seed=1337 \
    concat_binary.ratio_threshold=0.9
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import hydra
import numpy as np
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from PIL import Image
from tqdm import tqdm

# ----------------------
# Helpers
# ----------------------


def _squeeze_hw(arr: np.ndarray) -> np.ndarray:
    return arr[..., 0] if (arr.ndim == 3 and arr.shape[2] == 1) else arr


def count_cells_56(type_map: np.ndarray, inst_map: np.ndarray) -> Tuple[int, int]:
    """
    Count distinct instance IDs (inst_map>0) whose pixels are labeled 5 or 6.
    In your pipeline, type_map is uniform per instance, so this is safe.
    """
    inst = inst_map
    ids5 = np.unique(inst[(type_map == 5) & (inst > 0)])
    ids6 = np.unique(inst[(type_map == 6) & (inst > 0)])
    return int(len(ids5)), int(len(ids6))


def discover_pairs(root: Path) -> List[Tuple[Path, Path, str]]:
    """Return list of (png_path, npz_path, stem) for paired tiles by stem."""
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
    return np.asarray(img)


def extract_mask_from_npz(npz_path: Path, key_preference: str = 'type_map') -> np.ndarray:
    """Prefer `key_preference`; fall back to likely keys; return (H,W) uint8 labels."""
    data = np.load(npz_path, allow_pickle=True)
    key = key_preference if key_preference in data.files else None
    if key is None:
        for k in ['type_map', 'label', 'labels', 'mask', 'masks', 'arr_0', 'inst_map']:
            if k in data.files:
                key = k
                break
    if key is None:
        raise RuntimeError(f"No usable array key found in NPZ: {npz_path}")
    arr = np.array(data[key])
    arr = np.squeeze(arr)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise RuntimeError(f"Unsupported mask shape {arr.shape} in {npz_path}")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def infer_shapes(sample_pair: Tuple[Path, Path, str], mask_key: str) -> Tuple[int, int, int]:
    img = load_image(sample_pair[0])
    msk = extract_mask_from_npz(sample_pair[1], mask_key)
    H, W = img.shape[:2]
    if msk.shape != (H, W):
        raise RuntimeError(f"Shape mismatch for {sample_pair[2]}: image {img.shape} vs mask {msk.shape}")
    C_img = img.shape[2] if img.ndim == 3 else 1
    return H, W, C_img


def split_into_folds(stems: List[str], folds: int, seed: int) -> List[List[str]]:
    rng = np.random.default_rng(seed)
    order = np.array(stems)
    rng.shuffle(order)
    return [list(chunk) for chunk in np.array_split(order, folds)]


def stratified_split_by_major_12(
    stems: List[str],
    majority_12: Dict[str, int],  # 1 or 2 (or 0 if tie/none)
    folds: int,
    seed: int
) -> List[List[str]]:
    """Balance folds by which of {1,2} dominates among positives in each kept tile."""
    rng = np.random.default_rng(seed)
    buckets: Dict[int, List[str]] = {1: [], 2: [], 0: []}
    for s in stems:
        buckets.setdefault(int(majority_12.get(s, 0)), []).append(s)
    for k in buckets:
        rng.shuffle(buckets[k])
    fold_lists = [[] for _ in range(folds)]
    for k in [1, 2, 0]:
        items = buckets[k]
        for i, s in enumerate(items):
            fold_lists[i % folds].append(s)
    for lst in fold_lists:
        rng.shuffle(lst)
    return fold_lists


# ----------------------
# Core logic
# ----------------------

def tile_passes_ratio(msk: np.ndarray, threshold: float) -> Tuple[bool, int, int, int]:
    """
    Returns (keep, count5, count6, total_fg) with total_fg = count of labels != 0.
    keep is True if (count5+count6)/total_fg >= threshold and total_fg > 0.
    """
    fg = msk != 0
    total_fg = int(fg.sum())
    if total_fg == 0:
        return False, 0, 0, 0
    count5 = int((msk == 5).sum())
    count6 = int((msk == 6).sum())
    ratio = (count5 + count6) / total_fg
    return (ratio >= threshold), count5, count6, total_fg


def map_56_to_012(msk: np.ndarray) -> np.ndarray:
    """
    Map labels to ternary {0,1,2}:
      5 -> 1
      6 -> 2
      others -> 0
    Returns (H,W,1) uint8.
    """
    out = np.zeros_like(msk, dtype=np.uint8)
    out[msk == 5] = 1
    out[msk == 6] = 2
    return out[..., None]



def write_fold(
    fold_idx: int,
    stems: List[str],
    pair_map: Dict[str, Tuple[Path, Path]],
    out_root: Path,
    H: int,
    W: int,
    C_img: int,
    mask_key: str,
    stats_per_fold: Dict[str, Dict[str, int]],
    ratio_threshold: float,
    *,
    export_onehot: bool = True,
    out_channels: int = 6,
    channel_map: Optional[Dict[int, int]] = None,
) -> None:
    """
    Streamed writer:
      - First pass: decide which stems pass; accumulate cell counts.
      - Second pass: allocate memmaps of correct size and write tiles.
    """
    if channel_map is None:
        # Ternary {0,1,2} meaning {background, orig 5, orig 6} -> (6-channels) ch4, ch5
        channel_map = {1: 4, 2: 5}

    fold_dir = out_root / f"fold{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    img_path = fold_dir / "images.npy"
    msk_path = fold_dir / "masks.npy"

    kept: List[str] = []
    sum5_cells = 0
    sum6_cells = 0

    # -------- Pass 1: filter + stats --------
    for s in tqdm(stems, desc=f"fold{fold_idx} pass1/filter", unit="tile"):
        png_p, npz_p = pair_map[s]
        # type map (H,W)
        type_map = extract_mask_from_npz(npz_p, key_preference=mask_key)
        if type_map.shape != (H, W):
            continue

        if ratio_threshold > 0:
            keep, c5, c6, _ = tile_passes_ratio(type_map, threshold=ratio_threshold)
            if not keep:
                continue

        # cell counts via inst_map (ignore if missing)
        try:
            inst_map = extract_mask_from_npz(npz_p, key_preference="inst_map")
            inst_map = inst_map[..., 0] if (inst_map.ndim == 3 and inst_map.shape[2] == 1) else inst_map
            if inst_map.shape == (H, W):
                # count distinct instance ids with type 5 / 6
                ids5 = np.unique(inst_map[(type_map == 5) & (inst_map > 0)])
                ids6 = np.unique(inst_map[(type_map == 6) & (inst_map > 0)])
                sum5_cells += int(len(ids5))
                sum6_cells += int(len(ids6))
        except Exception:
            pass

        kept.append(s)

    N = len(kept)

    # Save stats right away (even if N=0)
    stats = {
        "tiles_kept": int(N),
        "cells_class_5": int(sum5_cells),
        "cells_class_6": int(sum6_cells),
    }
    stats_per_fold[f"fold{fold_idx}"] = stats
    with open(fold_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"[STATS] fold{fold_idx}: kept={N}, class5_cells={sum5_cells}, class6_cells={sum6_cells}")

    if N == 0:
        # Nothing to write; still produced stats.json
        return

    # -------- Pass 2: allocate + write --------
    mask_channels = out_channels if export_onehot else 1
    img_mm = np.lib.format.open_memmap(
        img_path, mode="w+", dtype=np.uint8, shape=(N, H, W, C_img)
    )
    msk_mm = np.lib.format.open_memmap(
        msk_path, mode="w+", dtype=np.uint8, shape=(N, H, W, mask_channels)
    )

    for i, s in enumerate(tqdm(kept, desc=f"fold{fold_idx} pass2/write", unit="tile")):
        png_p, npz_p = pair_map[s]
        img = load_image(png_p)  # (H,W,3)
        if img.shape[:2] != (H, W):
            raise RuntimeError(f"Image shape mismatch for {s}: {img.shape} vs {(H, W)}")
        if img.ndim == 2:
            img = img[..., None]
        if img.shape[2] != C_img:
            # enforce requested C_img (usually 3)
            if img.shape[2] == 1 and C_img == 3:
                img = np.repeat(img, 3, axis=2)
            elif img.shape[2] != C_img:
                raise RuntimeError(f"Inconsistent image channels for {s}: {img.shape[2]} vs {C_img}")
        img_mm[i] = img.astype(np.uint8)

        # type map (H,W), then map to {0,1,2}
        type_map = extract_mask_from_npz(npz_p, key_preference=mask_key)
        type_map = type_map[..., 0] if (type_map.ndim == 3 and type_map.shape[2] == 1) else type_map
        if type_map.shape != (H, W):
            raise RuntimeError(f"Mask shape mismatch for {s}: {type_map.shape} vs {(H, W)}")

        # ternary {0,1,2} where 1: orig 5, 2: orig 6
        m012 = np.zeros((H, W), dtype=np.uint8)
        m012[type_map == 5] = 1
        m012[type_map == 6] = 2

        if export_onehot:
            oh = np.zeros((H, W, out_channels), dtype=np.uint8)
            # fill only the configured channels
            for val, ch in channel_map.items():
                oh[..., int(ch)] = (m012 == int(val)).astype(np.uint8)
            msk_mm[i] = oh
        else:
            msk_mm[i] = m012[..., None]  # single channel

    # Flush
    del img_mm
    del msk_mm







# Global so helpers see the threshold
# RATIO_THRESHOLD_GLOBAL: float = 0.9

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # global RATIO_THRESHOLD_GLOBAL

    # Resolve against the original cwd (stable across Hydra run dirs)
    tiles_dir = Path(to_absolute_path(str(cfg.concat.tiles_dir)))
    out_root  = Path(to_absolute_path(str(cfg.concat.out_root)))

    folds            = int(cfg.concat.folds)
    seed             = int(cfg.concat.seed)
    mask_key         = str(cfg.concat.mask_key)
    stratify_by_major  = bool(cfg.concat.stratify_by_major)
    start_at_one  = bool(cfg.concat.fold_index_starts_at_one)
    RATIO_THRESHOLD_GLOBAL = float(cfg.concat.ratio_threshold)
    EXPORT_ONEHOT = bool(cfg.concat.export_onehot_for_cellvit)
    OUT_CHANNELS  = int(cfg.concat.num_classes)
    channel_map = dict(cfg.concat.channel_map)


    out_root.mkdir(parents=True, exist_ok=True)

    # rewritte rawmap depending on the format 
    if isinstance(channel_map, str):
        # e.g., "1:4,2:5"
        mapping = {}
        for kv in channel_map.split(','):
            if ':' in kv:
                k, v = kv.split(':', 1)
                mapping[int(k.strip())] = int(v.strip())
        CHANNEL_MAP = mapping if mapping else {1: 4, 2: 5}
    elif isinstance(channel_map, (list, tuple)):
        # e.g., [4,5] meaning {1->4, 2->5}
        CHANNEL_MAP = {i+1: int(ch) for i, ch in enumerate(channel_map)}
    else:
        CHANNEL_MAP = {int(k): int(v) for k, v in dict(channel_map).items()}

    # Make these available to write_fold via closure or put them global
    globals().update(dict(EXPORT_ONEHOT=EXPORT_ONEHOT,
                          OUT_CHANNELS=OUT_CHANNELS,
                          CHANNEL_MAP=CHANNEL_MAP))

    pairs = discover_pairs(tiles_dir)
    pair_map: Dict[str, Tuple[Path, Path]] = {s: (png, npz) for (png, npz, s) in pairs}
    stems_all = [s for (_, _, s) in pairs]

    # Infer shape from first pair
    H, W, C_img = infer_shapes(pairs[0], mask_key)
    print(f"[INFO] Image shape: (H={H}, W={W}, C={C_img}); mask key: '{mask_key}'")
    print(f"[INFO] Inclusion rule: >= {RATIO_THRESHOLD_GLOBAL*100:.1f}% of non-background pixels are class 5 or 6.")

    # Pre-scan: decide which tiles pass and (optionally) prepare stratification
    passing_stems: List[str] = []
    majority_12: Dict[str, int] = {}  # 1 or 2 (0 if tie/none)

    for s in tqdm(stems_all, desc="pre-scan", unit="tile"):
        m = extract_mask_from_npz(pair_map[s][1], mask_key)
        keep, c5, c6, _ = tile_passes_ratio(m, RATIO_THRESHOLD_GLOBAL)
        if not keep:
            continue
        passing_stems.append(s)
        if c5 == c6:
            maj = 0
        else:
            maj = 1 if c5 > c6 else 2
        majority_12[s] = maj

    if not passing_stems:
        print("[INFO] No tiles satisfied the ratio threshold. Nothing to write.")
        return

    # Split into folds
    if stratify_by_major:
        fold_stems = stratified_split_by_major_12(passing_stems, majority_12, folds, seed)
    else:
        fold_stems = split_into_folds(passing_stems, folds, seed)

    def fold_name(i: int) -> str:
        return f"fold{i+1}" if start_at_one else f"fold{i}"

    # Write each fold and collect stats
    stats_all: Dict[str, Dict[str, int]] = {}
    for fi, stems_i in enumerate(fold_stems):
        print(f"[INFO] Writing {fold_name(fi)} with {len(stems_i)} tiles …")
        write_fold(
            fi, stems_i, pair_map, out_root, H, W, C_img, mask_key,
            stats_per_fold=stats_all, ratio_threshold=RATIO_THRESHOLD_GLOBAL
        )
        print(f"[OK] {fold_name(fi)} saved to {out_root / fold_name(fi)}")

    # Save global summary
    with open(out_root / "summary_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats_all, f, indent=2)
    print("[DONE] All folds written.")
    print(json.dumps(stats_all, indent=2))



if __name__ == '__main__':
    main()
