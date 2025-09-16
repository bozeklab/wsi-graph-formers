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

from omegaconf import DictConfig
import hydra
from hydra.utils import to_absolute_path
from configs.schema import GraphConfig


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
      - type_map : uint8[tile, tile]   (per-pixel class 0..5; 0=background)

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



def process_one_slide(json_file: Path, wsi_dir: Path, out_root: Path,
                      tile_size: int, tmp_root: Path, no_bbox: bool) -> bool:
    """
    Convert a single slide (JSON + WSI) into PanNuke-style tiles,
    saving outputs inside a dedicated subfolder out_root/<stem>/.

    Returns
    -------
    bool
        True if tile(s) were processed or already existed; False if WSI is missing.
    """
    stem = json_file.stem
    wsi_path = find_matching_wsi(stem, wsi_dir)
    if wsi_path is None:
        log.warning(f"[skip] No matching WSI found in '{wsi_dir}' for basename '{stem}'.")
        return False

    # Subfolder for this slide
    slide_outdir = out_root / stem
    slide_outdir.mkdir(parents=True, exist_ok=True)

    # Skip if already processed: check for a sentinel tile (0,0)
    sample_tile = slide_outdir / f"{stem}_tile_0_0.npz"
    if sample_tile.exists():
        log.info(f"[skip] Output already exists for '{stem}' (found {sample_tile.relative_to(out_root)}).")
        return True

    # Read WSI size
    W, H = read_wsi_size_with_openslide(wsi_path)  # (W, H)

    # Per-slide shard temp dir
    shards_dir = tmp_root / f"shards_{stem}"
    shards_dir.mkdir(parents=True, exist_ok=True)

    # Stage A: stream + bin to per-tile shards
    stage_a_bin_instances_to_tiles(
        json_path=str(json_file),
        W=W, H=H, tile=tile_size,
        tmp_dir=shards_dir,
        assume_bbox_in_json=not no_bbox,
    )

    # Stage B: rasterize shards into this slide's output folder
    stage_b_rasterize_tiles(
        shards_dir=shards_dir,
        out_dir=slide_outdir,   # <<< write inside slide subfolder
        tile=tile_size,
        slide_stem=stem,
    )

    # Cleanup (best-effort)
    try:
        shards_dir.rmdir()
    except OSError:
        pass

    return True





@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):

    # Resolve against the original cwd (stable across Hydra run dirs)
    json_dir = Path(to_absolute_path(str(cfg.json_input_folder)))
    wsi_dir  = Path(to_absolute_path(str(cfg.wsi_input_folder)))
    out_dir  = Path(to_absolute_path(str(cfg.output_baselineset)))
    tmp_dir  = Path(to_absolute_path(str(cfg.tmp_dir)))

    tile_size = int(cfg.tile_size)
    no_bbox   = bool(cfg.no_bbox)

    # Ensure dirs exist
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Now Path.glob() works
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

    log.info(f"Done. Processed: {processed} | Skipped (missing WSI): {skipped}")

"""
Stream a full-WSI HoVer-Net JSON (stdlib json only) and emit PanNuke-style tiles (.npz),
auto-reading WSI width/height from a slide in another folder via OpenSlide.

Requirements:
pip install numpy opencv-python tqdm openslide-python openslide-bin

Outputs per tile (.npz):
- inst_map: (tile, tile) int32   (tile-local instance IDs 1..K)
- type_map: (tile, tile) uint8   (0..5; 0=background)
"""


if __name__ == "__main__":
    main()


