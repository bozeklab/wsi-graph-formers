"""
Lucas Sancéré 2025

Convert wsijson_to_trainset outputs (NPZ + PNG tiles)
into a HoVer-Net binary dataset (packed 5-channel .npy for extract_patches.py).

Inclusion rule
--------------
A tile is INCLUDED only if >= ratio_threshold of NON-BACKGROUND pixels
(label != 0) are of class 5 or 6. Default ratio_threshold = 0.90.

For included tiles:
  - labels {5,6} -> {1,2}
  - all other labels -> 0
  - instances not belonging to {5,6} are removed
  - remaining instances are reindexed to 1..K

Input directory (flat or nested):
  <tiles_root>/.../<tile>.png
  <tiles_root>/.../<tile>.npz   (expects keys 'inst_map' and 'type_map')

Outputs per fold:
  <out_root>/foldK/images/<tile>.png
  <out_root>/foldK/labels/<tile>.npy  (H,W,5) packed: [R,G,B,inst,type]
  <out_root>/summary_stats.json

Example:
  python generate_hovernetbinary.py \
    hvn_format.tiles_root=/path/to/tiles \
    hvn_format.hvn_format_output=/path/to/hovernet_bin_packed \
    hvn_format.folds=3 \
    hvn_format.seed=1337 \
    hvn_format.ratio_threshold=0.9
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import hydra
import numpy as np
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from PIL import Image
from tqdm import tqdm

# ----------------------
# Helpers
# ----------------------

def discover_pairs(root: Path) -> List[Tuple[Path, Path, str]]:
    """Return list of (png_path, npz_path, stem) for paired tiles by stem."""
    pngs = {p.stem: p for p in root.rglob("*.png")}
    npzs = {p.stem: p for p in root.rglob("*.npz")}
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
    return np.asarray(Image.open(path).convert("RGB"))


def load_npz_arrays(npz_path: Path, inst_key: str, type_key: str) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(npz_path, allow_pickle=False)
    if inst_key not in data.files or type_key not in data.files:
        raise KeyError(f"Missing keys in {npz_path.name}. Found: {data.files}")
    inst_map = np.array(data[inst_key])
    type_map = np.array(data[type_key])
    return inst_map, type_map


def tile_passes_ratio(type_map: np.ndarray, threshold: float) -> Tuple[bool, int, int, int]:
    """
    Returns (keep, count5, count6, total_fg) with total_fg = count of labels != 0.
    keep is True if (count5+count6)/total_fg >= threshold and total_fg > 0.
    """
    fg = type_map != 0
    total_fg = int(fg.sum())
    if total_fg == 0:
        return False, 0, 0, 0
    count5 = int((type_map == 5).sum())
    count6 = int((type_map == 6).sum())
    ratio = (count5 + count6) / total_fg
    return (ratio >= threshold), count5, count6, total_fg


def split_into_folds(stems: List[str], folds: int, seed: int) -> List[List[str]]:
    rng = np.random.default_rng(seed)
    order = np.array(stems)
    rng.shuffle(order)
    return [list(chunk) for chunk in np.array_split(order, folds)]


def stratified_split_by_major_56(
    stems: List[str],
    majority: Dict[str, int],  # 5 or 6 (or 0)
    folds: int,
    seed: int
) -> List[List[str]]:
    """Balance folds by which of {5,6} dominates among positives in each kept tile."""
    rng = np.random.default_rng(seed)
    buckets: Dict[int, List[str]] = {5: [], 6: [], 0: []}
    for s in stems:
        buckets.setdefault(int(majority.get(s, 0)), []).append(s)
    for k in buckets:
        rng.shuffle(buckets[k])

    fold_lists = [[] for _ in range(folds)]
    for k in [5, 6, 0]:
        items = buckets[k]
        for i, s in enumerate(items):
            fold_lists[i % folds].append(s)

    for lst in fold_lists:
        rng.shuffle(lst)
    return fold_lists


def filter_and_remap(inst_map: np.ndarray, type_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Keep only instances whose majority type is 5 or 6.
    Remap:
      5 -> 1
      6 -> 2
      others -> 0
    Reindex kept instances to 1..K.
    """
    inst_map = inst_map.astype(np.int32, copy=False)
    type_map = type_map.astype(np.uint8, copy=False)

    inst_new = np.zeros_like(inst_map, dtype=np.int32)
    type_new = np.zeros_like(type_map, dtype=np.uint8)

    ids = np.unique(inst_map)
    ids = ids[ids > 0]

    new_id = 0
    for oid in ids:
        mask = inst_map == oid
        if not mask.any():
            continue

        tvals = type_map[mask]
        t = int(np.bincount(tvals).argmax())  # robust majority type
        if t not in (5, 6):
            continue

        new_id += 1
        inst_new[mask] = new_id
        type_new[mask] = 1 if t == 5 else 2

    return inst_new, type_new


def pack_rgb_inst_type(
    rgb: np.ndarray,
    inst_map: np.ndarray,
    type_map: np.ndarray,
    dtype=np.int32
) -> np.ndarray:
    """
    Create packed tensor (H,W,5) with channels:
      [R, G, B, inst_map, type_map]
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"RGB must be (H,W,3); got {rgb.shape}")
    H, W = rgb.shape[:2]
    if inst_map.shape != (H, W) or type_map.shape != (H, W):
        raise ValueError(f"Shape mismatch rgb={rgb.shape}, inst={inst_map.shape}, type={type_map.shape}")

    packed = np.zeros((H, W, 5), dtype=dtype)
    packed[..., 0:3] = rgb.astype(dtype)
    packed[..., 3] = inst_map.astype(dtype)
    packed[..., 4] = type_map.astype(dtype)
    return packed


def export_fold(
    fold_idx: int,
    stems: List[str],
    pair_map: Dict[str, Tuple[Path, Path]],
    out_root: Path,
    ratio_threshold: float,
    inst_key: str,
    type_key: str,
    overwrite: bool,
    packed_dtype: str = "int32",
) -> Dict[str, int]:
    """
    For a list of stems, export PNG + packed NPY into out_root/fold{fold_idx}/.
    Returns stats dict.
    """
    fold_dir = out_root / f"fold{fold_idx}"
    img_dir = fold_dir / "images"
    lab_dir = fold_dir / "labels"  # keep folder name labels/
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)

    kept = 0
    skipped = 0
    cells_class_5 = 0
    cells_class_6 = 0

    dtype = np.dtype(packed_dtype)

    for s in tqdm(stems, desc=f"fold{fold_idx} export", unit="tile"):
        png_p, npz_p = pair_map[s]

        img_out = img_dir / f"{s}.png"
        npy_out = lab_dir / f"{s}.npy"  # labels/*.npy (packed)

        if (not overwrite) and img_out.exists() and npy_out.exists():
            kept += 1
            continue

        try:
            inst_map, type_map = load_npz_arrays(npz_p, inst_key, type_key)
        except Exception:
            skipped += 1
            continue

        # Ratio gate only when threshold > 0
        if ratio_threshold > 0:
            keep, _, _, _ = tile_passes_ratio(type_map, ratio_threshold)
            if not keep:
                skipped += 1
                continue

        inst_new, type_new = filter_and_remap(inst_map, type_map)

        # Only enforce "must have 5/6 instances" when threshold > 0
        if ratio_threshold > 0 and inst_new.max() == 0:
            skipped += 1
            continue


        # count instances per class (type_new is {0,1,2})
        ids1 = np.unique(inst_new[(type_new == 1) & (inst_new > 0)])
        ids2 = np.unique(inst_new[(type_new == 2) & (inst_new > 0)])
        cells_class_5 += int(len(ids1))
        cells_class_6 += int(len(ids2))

        # write image + packed label
        rgb = load_image(png_p)
        Image.fromarray(rgb).save(img_out)

        packed = pack_rgb_inst_type(rgb, inst_new, type_new, dtype=dtype)
        np.save(npy_out, packed)

        kept += 1

    stats = {
        "tiles_kept": int(kept),
        "tiles_skipped": int(skipped),
        "cells_class_5": int(cells_class_5),
        "cells_class_6": int(cells_class_6),
    }
    with open(fold_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"[STATS] fold{fold_idx}: kept={kept}, skipped={skipped}, "
          f"class5_cells={cells_class_5}, class6_cells={cells_class_6}")

    return stats


# ----------------------
# Core logic
# ----------------------

@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    tiles_root = Path(to_absolute_path(str(cfg.hvn_format.tiles_root)))
    hvn_format_output = Path(to_absolute_path(str(cfg.hvn_format.hvn_format_output)))

    folds = int(cfg.hvn_format.folds)
    seed = int(cfg.hvn_format.seed)
    ratio_threshold = float(cfg.hvn_format.ratio_threshold)
    stratify_by_major = bool(cfg.hvn_format.stratify_by_major)

    overwrite = bool(getattr(cfg.hvn_format, "overwrite", False))
    packed_dtype = str(getattr(cfg.hvn_format, "packed_dtype", "int32"))

    inst_key = "inst_map"
    type_key = "type_map"

    hvn_format_output.mkdir(parents=True, exist_ok=True)

    pairs = discover_pairs(tiles_root)
    pair_map: Dict[str, Tuple[Path, Path]] = {s: (png, npz) for (png, npz, s) in pairs}
    stems_all = [s for (_, _, s) in pairs]

    print(f"[INFO] Found {len(stems_all)} paired tiles under: {tiles_root}")
    print(f"[INFO] Inclusion rule: >= {ratio_threshold*100:.1f}% of non-background pixels are class 5 or 6.")

    # Pre-scan: decide which tiles pass and (optionally) prepare stratification
    passing_stems: List[str] = []
    majority: Dict[str, int] = {}  # 5 or 6 (or 0)

    for s in tqdm(stems_all, desc="pre-scan", unit="tile"):
        _, npz_p = pair_map[s]
        try:
            _, type_map = load_npz_arrays(npz_p, inst_key, type_key)
        except Exception:
            continue

        keep, c5, c6, _ = tile_passes_ratio(type_map, ratio_threshold)
        if not keep:
            continue

        passing_stems.append(s)

        # majority among positives (5 vs 6) for stratification
        if c5 == c6:
            maj = 0
        else:
            maj = 5 if c5 > c6 else 6
        majority[s] = maj

    if not passing_stems:
        print("[INFO] No tiles satisfied the ratio threshold. Nothing to export.")
        return

    # Split into folds
    if stratify_by_major:
        fold_stems = stratified_split_by_major_56(passing_stems, majority, folds, seed)
    else:
        fold_stems = split_into_folds(passing_stems, folds, seed)

    # Export each fold
    stats_all: Dict[str, Dict[str, int]] = {}
    for fi, stems_i in enumerate(fold_stems):
        print(f"[INFO] Writing fold{fi} with {len(stems_i)} tiles …")
        stats_all[f"fold{fi}"] = export_fold(
            fold_idx=fi,
            stems=stems_i,
            pair_map=pair_map,
            out_root=hvn_format_output,
            ratio_threshold=ratio_threshold,
            inst_key=inst_key,
            type_key=type_key,
            overwrite=overwrite,
            packed_dtype=packed_dtype,
        )
        print(f"[OK] fold{fi} saved to {hvn_format_output / f'fold{fi}'}")

    # Save global summary
    with open(hvn_format_output / "summary_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats_all, f, indent=2)

    print("[DONE] All folds written.")
    print(json.dumps(stats_all, indent=2))


if __name__ == "__main__":
    main()
