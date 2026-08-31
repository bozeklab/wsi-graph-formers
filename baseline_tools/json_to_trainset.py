"""
Lucas Sancéré 2025
"""

import os
import json
import math
import argparse
import tempfile
from pathlib import Path
from typing import Dict, Any, Tuple, Iterable, Generator, Optional

import numpy as np
import cv2
from tqdm import tqdm
import openslide  # provides slide.dimensions == (W, H) at level 0
from collections import OrderedDict
import logging
import tifffile

from omegaconf import DictConfig
import hydra
from hydra.utils import to_absolute_path


log = logging.getLogger(__name__)


#### geometry helpers 

def bbox_from_contour(contour_np: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Compute an axis-aligned bounding box from a polygon.

    Parameters
    ----------
    contour_np : np.ndarray
        Array of shape [N, 2] with (x, y) polygon vertices (float or int).

    Returns
    -------
    (float, float, float, float)
        (xmin, ymin, xmax, ymax) of the polygon.
    """
    xmin = float(contour_np[:, 0].min())
    xmax = float(contour_np[:, 0].max())
    ymin = float(contour_np[:, 1].min())
    ymax = float(contour_np[:, 1].max())
    return xmin, ymin, xmax, ymax


def tiles_overlapping_bbox(xmin: float, ymin: float, xmax: float, ymax: float,
                           tile: int, W: int, H: int) -> Iterable[Tuple[int, int]]:
    """
    Yield tile indices that overlap a bounding box on a WSI grid.

    Parameters
    ----------
    xmin, ymin, xmax, ymax : float
        Bounding box in level-0 pixel coordinates.
    tile : int
        Tile size (both width and height), e.g., 256 for PanNuke-style tiles.
    W, H : int
        Whole-slide dimensions in pixels at level 0 (width W, height H).

    """
    ix0 = max(int(math.floor(xmin / tile)), 0)
    iy0 = max(int(math.floor(ymin / tile)), 0)
    ix1 = min(int(math.floor(xmax / tile)), max((W - 1) // tile, 0))
    iy1 = min(int(math.floor(ymax / tile)), max((H - 1) // tile, 0))
    for ty in range(iy0, iy1 + 1):
        for tx in range(ix0, ix1 + 1):
            yield tx, ty


def clip_and_shift_contour(contour: np.ndarray, x0: int, y0: int, tile: int) -> np.ndarray:
    """
    Translate a global-XY polygon into tile-local coordinates and clip to tile bounds.

    Parameters
    ----------
    contour : np.ndarray
        Array of shape [N, 2] with global (x, y) vertices (float or int).
    x0, y0 : int
        Top-left global pixel of the tile (tx*tile, ty*tile).
    tile : int
        Tile size (width = height = tile).

    Returns
    -------
    np.ndarray
        Array of shape [N, 1, 2] int32 suitable for `cv2.fillPoly`.
        Points are clipped to [0, tile-1] in both axes.
    """
    pts = contour.astype(np.float64)
    pts[:, 0] -= x0
    pts[:, 1] -= y0
    pts[:, 0] = np.clip(pts[:, 0], 0, tile - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, tile - 1)
    return pts.astype(np.int32).reshape(-1, 1, 2)






#### stdlib JSON: stream top-level {key: value} 

def iter_top_level_kv(path: str) -> Generator[Tuple[str, Any], None, None]:
    """
    Stream (key, value) pairs from a HUGE JSON that is a top-level dict, without `json.load`.

    This avoids materializing gigabyte-scale JSONs in memory by scanning the file with a
    small state machine that respects strings, escapes, and balanced { } / [ ].

    Parameters
    ----------
    path : str
        Path to a JSON file whose top level is a dictionary:
        { "<inst_id>": { ... }, "<inst_id>": { ... }, ... }.
    """
    with open(path, 'rt', encoding='utf-8') as f:
        # seek '{'
        c = f.read(1)
        while c and c.isspace():
            c = f.read(1)
        if c != '{':
            raise ValueError("Top-level JSON must start with '{'")

        def read_string(initial_quote: str) -> str:
            s = [initial_quote]
            esc = False
            while True:
                ch = f.read(1)
                if not ch:
                    raise ValueError("Unterminated string")
                s.append(ch)
                if esc:
                    esc = False
                    continue
                if ch == '\\':
                    esc = True
                    continue
                if ch == initial_quote:
                    return ''.join(s)

        first_pair = True
        while True:
            # skip ws/comma
            c = f.read(1)
            while c and c.isspace():
                c = f.read(1)
            if not c:
                raise ValueError("Unexpected EOF in object")
            if c == '}':
                break
            if not first_pair:
                if c != ',':
                    raise ValueError(f"Expected ',' between pairs, got {repr(c)}")
                c = f.read(1)
                while c and c.isspace():
                    c = f.read(1)
                if c == '}':
                    break
            first_pair = False

            # key
            if c not in ('"', "'"):
                raise ValueError("JSON object keys must be strings")
            key_text = read_string(c)
            key = json.loads(key_text)

            # colon
            c = f.read(1)
            while c and c.isspace():
                c = f.read(1)
            if c != ':':
                raise ValueError("Expected ':' after key")

            # value start
            c = f.read(1)
            while c and c.isspace():
                c = f.read(1)
            if not c:
                raise ValueError("Unexpected EOF before value")

            # primitive (number/true/false/null)
            if c in '-0123456789tfn':
                buf = [c]
                while True:
                    c2 = f.read(1)
                    if not c2 or c2 in ',}]' or c2.isspace():
                        if c2:
                            f.seek(f.tell() - 1)
                        break
                    buf.append(c2)
                yield key, json.loads(''.join(buf))
                continue

            # string value
            if c in ('"', "'"):
                yield key, json.loads(read_string(c))
                continue

            # object/array
            buf = [c]
            depth_brace = 1 if c == '{' else 0
            depth_bracket = 1 if c == '[' else 0
            in_str = False
            esc = False
            while True:
                c2 = f.read(1)
                if not c2:
                    raise ValueError("Unexpected EOF in value")
                buf.append(c2)
                if in_str:
                    if esc:
                        esc = False
                    elif c2 == '\\':
                        esc = True
                    elif c2 == '"':
                        in_str = False
                    continue
                if c2 == '"':
                    in_str = True
                elif c2 == '{':
                    depth_brace += 1
                elif c2 == '}':
                    depth_brace -= 1
                    if depth_brace == 0 and depth_bracket == 0:
                        break
                elif c2 == '[':
                    depth_bracket += 1
                elif c2 == ']':
                    depth_bracket -= 1
                    if depth_bracket == 0 and depth_brace == 0:
                        break
            yield key, json.loads(''.join(buf))


#### WSI discovery & size via OpenSlide

WSI_EXTS = [
    ".svs", ".tif", ".tiff", ".ndpi", ".scn", ".mrxs", ".vms", ".vmu",
    ".svslide", ".bif", ".czi", ".dcm", ".qptiff"
]

def find_matching_wsi(stem: str, wsi_dir: Path) -> Optional[Path]:
    """
    Find a WSI in `wsi_dir` that shares the same basename as the JSON.

    Parameters
    ----------
    stem : str
        Basename (filename without extension), e.g., 'SlideA'.
    wsi_dir : pathlib.Path
        Folder containing WSIs.

    Returns
    -------
    Optional[pathlib.Path]
        Matching slide path if found (tries known WSI extensions, then any match),
        else None.
    """
    for ext in WSI_EXTS:
        p = wsi_dir / f"{stem}{ext}"
        if p.exists():
            return p
    matches = list(wsi_dir.glob(stem + ".*"))
    return matches[0] if matches else None


def read_wsi_size_with_openslide(path: Path) -> Tuple[int, int]:
    """
    Read the full-resolution whole-slide size (width, height) using OpenSlide.

    Parameters
    ----------
    path : pathlib.Path
        Path to the WSI file.

    Returns
    -------
    (int, int)
        (W, H) pixel dimensions at level 0.

    """
    slide = openslide.OpenSlide(str(path))
    W, H = slide.dimensions  # (W, H) at level 0
    slide.close()
    return int(W), int(H)


def read_flat_tiff_size(path: Path) -> Tuple[int, int]:
    """
    Read full-resolution image size (W, H) from a non-pyramidal TIFF file.

    Parameters
    ----------
    path : pathlib.Path
        Path to a non-pyramidal .tif file.

    Returns
    -------
    (int, int)
        (W, H) pixel dimensions.
    """
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]  # single full-resolution image
        H, W = page.shape[:2]

    return int(W), int(H)



def stage_a_bin_instances_to_tiles(
    json_path: str,
    W: int,
    H: int,
    tile: int,
    tmp_dir: Path,
    assume_bbox_in_json: bool = True,
    min_polygon_points: int = 3,
    max_open_shards: int = 512,   # NEW: cap concurrently-open shard files
):
    """
    Stream the HoVer-Net JSON and write per-tile JSONL shards:
    shard_{ty}_{tx}.jsonl with lines: {"id": int, "type": int, "contour": [[x,y], ...]}

    Parameters
    ----------
    ...
    max_open_shards : int
        Max number of shard files kept open at once. When exceeded, the least
        recently used file handle is closed to stay within OS limits.
    """
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # LRU of open file handles: key=(tx,ty) -> file object
    open_fhs: "OrderedDict[Tuple[int,int], Any]" = OrderedDict()

    def get_writer(tx: int, ty: int):
        """Return an open file handle for (tx,ty), respecting the LRU cap."""
        key = (tx, ty)
        fh = open_fhs.get(key)
        if fh is not None:
            # mark as most recently used
            open_fhs.move_to_end(key, last=True)
            return fh

        # Need a new handle
        shard_path = tmp_dir / f"shard_{ty}_{tx}.jsonl"
        fh = open(shard_path, 'a', encoding='utf-8')

        open_fhs[key] = fh
        open_fhs.move_to_end(key, last=True)

        # Enforce cap
        if len(open_fhs) > max_open_shards:
            old_key, old_fh = open_fhs.popitem(last=False)  # least recently used
            try:
                old_fh.flush()
            finally:
                old_fh.close()
        return fh

    try:
        for inst_id_str, inst in tqdm(iter_top_level_kv(json_path),
                                      desc=f"[{Path(json_path).stem}] streaming instances",
                                      unit="nuc"):
            # robust id
            try:
                inst_id = int(inst_id_str)
            except Exception:
                try:
                    inst_id = int(str(inst_id_str))
                except Exception:
                    continue

            contour = inst.get('contour')
            if not contour or len(contour) < min_polygon_points:
                continue
            inst_type = int(inst.get('type', 0))

            if assume_bbox_in_json and 'bbox' in inst:
                xmin, ymin = inst['bbox'][0]
                xmax, ymax = inst['bbox'][1]
            else:
                contour_np = np.asarray(contour, dtype=np.float64)
                xmin, ymin, xmax, ymax = bbox_from_contour(contour_np)

            # Write one record into every overlapping tile's shard
            rec = json.dumps({"id": inst_id, "type": inst_type, "contour": contour}) + "\n"
            for tx, ty in tiles_overlapping_bbox(xmin, ymin, xmax, ymax, tile, W, H):
                fh = get_writer(tx, ty)
                fh.write(rec)
    finally:
        # Close any remaining open files
        for fh in open_fhs.values():
            try:
                fh.flush()
            finally:
                fh.close()




def stage_b_rasterize_tiles(
    shards_dir: Path,
    out_dir: Path,
    tile: int,
    slide_stem: str,
    skip_empty: bool = True,
):
    """
    Rasterize per-tile shards into PanNuke-style `.npz` files.

    For each `shard_{ty}_{tx}.jsonl`, this creates `<stem>_tile_{ty}_{tx}.npz`
    with:
      - inst_map : int32[tile, tile]   (tile-local instance ids 1..K)
      - type_map : uint8[tile, tile]   (per-pixel class 0..6; 0=background)

    Parameters
    ----------
    shards_dir : pathlib.Path
        Folder containing the per-tile shard JSONL files.
    out_dir : pathlib.Path
        Destination folder for `.npz` tiles.
    tile : int
        Tile size (e.g., 256).
    slide_stem : str
        Basename of the slide (used in output filenames).
    skip_empty : bool, default True
        If True, tiles with no instances are discarded.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for shard_path in tqdm(sorted(shards_dir.glob("shard_*.jsonl")),
                           desc=f"[{slide_stem}] rasterizing tiles"):
        name = shard_path.stem  # shard_{ty}_{tx}
        _, ty_str, tx_str = name.split('_')
        tx, ty = int(tx_str), int(ty_str)
        x0, y0 = tx * tile, ty * tile

        inst_map = np.zeros((tile, tile), dtype=np.int32)
        type_map = np.zeros((tile, tile), dtype=np.uint8)

        local_id = 0
        id_map: Dict[int, int] = {}

        with open(shard_path, 'r', encoding='utf-8') as fh:
            for line in fh:
                rec = json.loads(line)
                oid = int(rec["id"])
                t = int(rec["type"])
                contour = np.asarray(rec["contour"], dtype=np.float64)
                pts = clip_and_shift_contour(contour, x0, y0, tile)
                if pts.size == 0:
                    continue
                if oid not in id_map:
                    local_id += 1
                    id_map[oid] = local_id
                lid = id_map[oid]
                # filled polygon → instance assignment
                cv2.fillPoly(inst_map, [pts], color=int(lid))
                # assign class where this instance id is present
                type_map[inst_map == lid] = t

        if skip_empty and inst_map.max() == 0:
            shard_path.unlink(missing_ok=True)
            continue

        out_path = out_dir / f"{slide_stem}_tile_{ty}_{tx}.npz"
        np.savez_compressed(out_path, inst_map=inst_map, type_map=type_map)
        shard_path.unlink(missing_ok=True)



def stage_c_export_png_tiles(
    wsi_path: Path,
    tiles_dir: Path,
    tile: int,
    slide_stem: str,
    level: int = 0,
    overwrite: bool = False,
):
    """
    Export RGB PNG patches from the WSI at the exact (ty, tx) tiles for which
    corresponding NPZ tiles (<stem>_tile_{ty}_{tx}.npz) already exist.

    This guarantees the PNGs align 1:1 with the PanNuke-style NPZ tiles produced
    in stage B.

    Parameters
    ----------
    wsi_path : pathlib.Path
        Path to the whole-slide image (WSI).
    tiles_dir : pathlib.Path
        Directory containing the NPZ tiles (and where PNGs will be written).
    tile : int
        Tile size in pixels (width = height = tile).
    slide_stem : str
        Basename used for tile filenames (e.g., 'SlideA').
    level : int, default 0
        OpenSlide level to read from (0 = full resolution). Use 0 if NPZs were
        generated from level-0 coordinates (the usual case).
    overwrite : bool, default False
        If False, skip writing PNG if it already exists.
    """
    tiles_dir.mkdir(parents=True, exist_ok=True)

    # Find NPZ tiles to mirror as PNGs
    npz_files = sorted(tiles_dir.glob(f"{slide_stem}_tile_*_*.npz"))
    if not npz_files:
        log.warning(f"[{slide_stem}] No NPZ tiles found in {tiles_dir} — nothing to export as PNG.")
        return

    # Open the slide once
    slide = openslide.OpenSlide(str(wsi_path))

    try:
        # Validate the requested level
        if level < 0 or level >= slide.level_count:
            raise ValueError(f"Requested level {level} out of range [0..{slide.level_count-1}]")

        # Determine downsample for the chosen level to compute correct read sizes
        # (For level=0 this will be 1.0)
        downsample = float(slide.level_downsamples[level])

        for npz_path in tqdm(npz_files, desc=f"[{slide_stem}] exporting PNG tiles"):
            # Parse ty, tx from "<stem>_tile_{ty}_{tx}.npz"
            name = npz_path.stem  # "<stem>_tile_{ty}_{tx}"
            try:
                _, _, ty_str, tx_str = name.split('_')  # ["<stem>", "tile", "{ty}", "{tx}"]
            except ValueError:
                # Fallback robust parse
                parts = name.split('_')
                if len(parts) < 4 or parts[-3] != "tile":
                    log.warning(f"[{slide_stem}] Unexpected tile filename format: {name}")
                    continue
                ty_str, tx_str = parts[-2], parts[-1]

            ty, tx = int(ty_str), int(tx_str)
            x0, y0 = tx * tile, ty * tile

            png_path = tiles_dir / f"{slide_stem}_tile_{ty}_{tx}.png"
            if (not overwrite) and png_path.exists():
                continue

            # read_region expects level-0 coordinates; size is in pixels at the requested level.
            # When reading at 'level', the returned image size is exactly (tile, tile) if we pass
            # size=(tile, tile) and the lib will internally scale based on 'level'.
            #
            # However, NPZ tiles were created from level-0 coordinates with size 'tile'.
            # To ensure perfect alignment, we read at level=0 with size=(tile, tile) by default.
            # If a non-zero 'level' is explicitly requested, we still pass (tile, tile) so that
            # the patch corresponds to the same field of view but sampled at that pyramid level.
            try:
                region = slide.read_region((x0, y0), level, (tile, tile))  # PIL RGBA
            except Exception as e:
                log.error(f"[{slide_stem}] read_region failed for (tx={tx}, ty={ty}) at level {level}: {e}")
                continue

            # Convert RGBA -> BGR for OpenCV, drop alpha
            rgba = np.asarray(region)  # H x W x 4
            if rgba.ndim != 3 or rgba.shape[2] != 4:
                log.error(f"[{slide_stem}] Unexpected region shape at (tx={tx}, ty={ty}): {rgba.shape}")
                continue

            bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)

            # Write PNG
            ok = cv2.imwrite(str(png_path), bgr)
            if not ok:
                log.error(f"[{slide_stem}] Failed to write {png_path}")

    finally:
        slide.close()


def stage_c_export_png_tiles_flat_tiff(
    wsi_path: Path,
    tiles_dir: Path,
    tile: int,
    slide_stem: str,
    level: int = 0,
    overwrite: bool = False,
):
    """
    Export RGB PNG patches from a NON-PYRAMIDAL .tif at the exact (ty, tx) tiles for which
    corresponding NPZ tiles (<stem>_tile_{ty}_{tx}.npz) already exist.

    This is the flat-TIFF equivalent of the OpenSlide-based exporter.

    Notes
    -----
    - Flat TIFFs have only one resolution level, so `level` must be 0.
    - Uses tifffile.memmap() to avoid loading the entire TIFF into RAM.
    """
    tiles_dir.mkdir(parents=True, exist_ok=True)

    # Find NPZ tiles to mirror as PNGs
    npz_files = sorted(tiles_dir.glob(f"{slide_stem}_tile_*_*.npz"))
    if not npz_files:
        log.warning(f"[{slide_stem}] No NPZ tiles found in {tiles_dir} — nothing to export as PNG.")
        return

    # Flat TIFF: enforce level=0
    if level != 0:
        raise ValueError(f"[{slide_stem}] Non-pyramidal TIFF supports only level=0 (got level={level}).")

    # Memory-mapped access to the full-res TIFF (does not load full image)
    try:
        img = tifffile.memmap(wsi_path)  # shape: (H, W) or (H, W, C)
    except Exception as e:
        raise RuntimeError(f"[{slide_stem}] Failed to memmap TIFF {wsi_path}: {e}") from e

    # Optional: basic sanity check
    if img.ndim not in (2, 3):
        raise ValueError(f"[{slide_stem}] Unexpected TIFF array shape: {img.shape}")

    H, W = img.shape[:2]

    for npz_path in tqdm(npz_files, desc=f"[{slide_stem}] exporting PNG tiles (flat TIFF)"):
        # Parse ty, tx from "<stem>_tile_{ty}_{tx}.npz"
        name = npz_path.stem
        try:
            _, _, ty_str, tx_str = name.split('_')
        except ValueError:
            parts = name.split('_')
            if len(parts) < 4 or parts[-3] != "tile":
                log.warning(f"[{slide_stem}] Unexpected tile filename format: {name}")
                continue
            ty_str, tx_str = parts[-2], parts[-1]

        ty, tx = int(ty_str), int(tx_str)
        x0, y0 = tx * tile, ty * tile

        png_path = tiles_dir / f"{slide_stem}_tile_{ty}_{tx}.png"
        if (not overwrite) and png_path.exists():
            continue

        # Bounds check (optional but helps avoid weird edge tiles)
        x1, y1 = x0 + tile, y0 + tile
        if x0 < 0 or y0 < 0 or x1 > W or y1 > H:
            log.error(
                f"[{slide_stem}] Tile (tx={tx}, ty={ty}) "
                f"patch=({x0}:{x1}, {y0}:{y1}) img_size=({W}x{H})"
            )
            continue

        # Extract patch (no scaling, since level=0 only)
        patch = img[y0:y1, x0:x1]

        # Convert to uint8 if needed (common for 16-bit TIFFs)
        if patch.dtype != np.uint8:
            # simple linear scaling to 0..255
            maxv = np.iinfo(patch.dtype).max if np.issubdtype(patch.dtype, np.integer) else float(patch.max() or 1.0)
            patch = (patch.astype(np.float32) / float(maxv) * 255.0).clip(0, 255).astype(np.uint8)

        # Make RGBA to match your OpenSlide path
        if patch.ndim == 2:
            # grayscale -> RGB
            patch = np.stack([patch, patch, patch], axis=-1)

        if patch.shape[2] == 3:
            alpha = np.full((tile, tile, 1), 255, dtype=np.uint8)
            rgba = np.concatenate([patch, alpha], axis=2)
        elif patch.shape[2] == 4:
            rgba = patch
        else:
            log.error(f"[{slide_stem}] Unsupported channel count in patch: {patch.shape}")
            continue

        # Convert RGBA -> BGR for OpenCV, drop alpha
        bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)

        ok = cv2.imwrite(str(png_path), bgr)
        if not ok:
            log.error(f"[{slide_stem}] Failed to write {png_path}")




def process_one_slide(
    json_file: Path,
    wsi_dir: Path,
    out_root: Path,
    tile_size: int,
    tmp_root: Path,
    no_bbox: bool,
    tilejson: bool,
    generate_npz: bool = True,
    generate_png: bool = False,
    png_level: int = 0,
    png_overwrite: bool = False,
) -> bool:
    """
    Convert a single slide (JSON + WSI) into PanNuke-style tiles and/or PNG patches,
    saving outputs inside a dedicated subfolder out_root/<stem>/.

    Parameters
    ----------
    generate_npz : bool, default True
        If True, run Stage A+B to produce NPZ tiles.
    generate_png : bool, default False
        If True, export PNG tiles (Stage C) for every NPZ tile present
        in the slide's output folder.
    png_level : int, default 0
        OpenSlide level for PNG export (Stage C).
    png_overwrite : bool, default False
        If False, existing PNGs will be skipped.

    Returns
    -------
    bool
        True if work was done or outputs already existed; False if WSI is missing.
    """
    stem = json_file.stem
    wsi_path = find_matching_wsi(stem, wsi_dir)
    if wsi_path is None:
        log.warning(f"[skip] No matching WSI found in '{wsi_dir}' for basename '{stem}'.")
        return False

    slide_outdir = out_root / stem
    slide_outdir.mkdir(parents=True, exist_ok=True)

    # Sentinel files to detect existing work
    sample_npz = slide_outdir / f"{stem}_tile_0_0.npz"
    sample_png = slide_outdir / f"{stem}_tile_0_0.png"

    did_anything = False

    # === Stage A+B: NPZ generation ===
    if generate_npz:
        if sample_npz.exists():
            log.info(f"[skip] NPZ already exists for '{stem}' (found {sample_npz.relative_to(out_root)}).")
        else:
            # Read WSI (W, H) and run staging
            if tilejson:
                W, H = read_flat_tiff_size(wsi_path)
            else:
                W, H = read_wsi_size_with_openslide(wsi_path)
            shards_dir = tmp_root / f"shards_{stem}"
            shards_dir.mkdir(parents=True, exist_ok=True)

            stage_a_bin_instances_to_tiles(
                json_path=str(json_file),
                W=W, H=H, tile=tile_size,
                tmp_dir=shards_dir,
                assume_bbox_in_json=not no_bbox,
            )

            stage_b_rasterize_tiles(
                shards_dir=shards_dir,
                out_dir=slide_outdir,
                tile=tile_size,
                slide_stem=stem,
            )

            # Cleanup shards (best-effort)
            try:
                shards_dir.rmdir()
            except OSError:
                pass

            did_anything = True

    # === Stage C: PNG export ===
    if generate_png:
        # Export PNGs for every NPZ tile present in slide_outdir
        # (works whether NPZs were just created above or already existed)
        npz_tiles = list(slide_outdir.glob(f"{stem}_tile_*_*.npz"))
        if not npz_tiles:
            log.warning(f"[{stem}] No NPZ tiles found to mirror as PNGs in {slide_outdir}. Skipping PNG export.")
        else:
            if tilejson:
                stage_c_export_png_tiles_flat_tiff(
                    wsi_path=wsi_path,
                    tiles_dir=slide_outdir,
                    tile=tile_size,
                    slide_stem=stem,
                    level=png_level,
                    overwrite=png_overwrite,
                )
                did_anything = True
            else:
                stage_c_export_png_tiles(
                    wsi_path=wsi_path,
                    tiles_dir=slide_outdir,
                    tile=tile_size,
                    slide_stem=stem,
                    level=png_level,
                    overwrite=png_overwrite,
                )
                did_anything = True

    return True if (did_anything or sample_npz.exists() or sample_png.exists()) else False






@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):

    # Resolve against the original cwd (stable across Hydra run dirs)
    json_dir = Path(to_absolute_path(str(cfg.json_input_folder)))
    wsi_dir  = Path(to_absolute_path(str(cfg.wsi_input_folder)))
    out_dir  = Path(to_absolute_path(str(cfg.output_baselineset)))
    tmp_dir  = Path(to_absolute_path(str(cfg.tmp_dir)))

    tile_size = int(cfg.tile_size)
    no_bbox   = bool(cfg.no_bbox)
    tilejson = bool(cfg.tilejson)

    generate_npz = bool(cfg.generate_npz)
    generate_png = bool(cfg.generate_png)
    png_level  = int(cfg.png_level)
    png_overwrite = bool(cfg.png_overwrite) 

    # Ensure dirs exist
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Discover JSON files
    json_files = sorted(json_dir.glob("*.json"))
    if not json_files:
        log.error(f"No JSON files found in {json_dir}")
        return
    # if not json_files:
    #     raise RuntimeError(f"No JSON files found in {json_dir}")

    processed = 0
    skipped   = 0

    for jf in json_files:
        ok = process_one_slide(
            json_file=jf,
            wsi_dir=wsi_dir,
            out_root=out_dir,
            tile_size=tile_size,
            tmp_root=tmp_dir,
            no_bbox=no_bbox,
            tilejson=tilejson,
            generate_npz=generate_npz,
            generate_png=generate_png,
            png_level=png_level,
            png_overwrite=png_overwrite,
        )
        if ok:
            processed += 1
        else:
            skipped += 1

    # best-effort tmp cleanup (will fail if non-empty; that's fine)
    try:
        tmp_dir.rmdir()
    except OSError:
        pass

    log.info(
        f"Done. Processed: {processed} | Skipped (missing WSI): {skipped} | "
        f"NPZ: {'on' if generate_npz else 'off'} | PNG: {'on' if generate_png else 'off'} "
        f"(level={png_level}, overwrite={png_overwrite})"
    )

"""
Stream a full-WSI HoVer-Net JSON (stdlib json only) and emit PanNuke-style tiles (.npz),
auto-reading WSI width/height from a slide in another folder via OpenSlide. Also can generate 
image files as pngs corresponding to the npz file generated. 

Requirements:
pip install numpy opencv-python tqdm openslide-python openslide-bin

Outputs per tile (.npz):
- inst_map: (tile, tile) int32   (tile-local instance IDs 1..K)
- type_map: (tile, tile) uint8   (0..6; 0=background)
"""


if __name__ == "__main__":
    main()


