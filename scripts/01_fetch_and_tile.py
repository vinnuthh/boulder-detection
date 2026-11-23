#!/usr/bin/env python
"""Stage 2 — fetch a real LROC NAC-derived GeoTIFF and cut 512px tiles.

Source priority
---------------
1. The configured Apollo 17 LROC NAC 0.5 m/px orthomosaic (HTTP range reads).
2. Any other GeoTIFF URL listed in ``tiling.nac_urls``.
3. Procedural lunar terrain fallback (configurable), so the pipeline never
   stops a demo because a server is down.

All output tiles are written in a local lunar Azimuthal Equidistant CRS in
metres, so ``rasterio.open(tile).transform.a`` is the real GSD in m/px and
the viewer can use a simple metric map.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
import rasterio
import requests
from ruamel.yaml import YAML
yaml = YAML(typ="safe")
from rasterio.crs import CRS
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject, transform as warp_transform, transform_bounds
from rasterio.windows import Window
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lunar_synth import render_scene, to_uint8  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
MOON_RADIUS_M = 1_737_400.0
GEO_CRS = CRS.from_proj4(f"+proj=longlat +R={MOON_RADIUS_M} +no_defs")

os.environ.setdefault("GDAL_HTTP_TIMEOUT", "60")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "3")
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS",
                      ".tif,.tiff,.TIF,.TIFF,.img,.IMG,.jp2,.JP2")


def load_config() -> dict:
    with open(ROOT / "config.yaml", "r", encoding="utf-8") as fh:
        return yaml.load(fh)


def log(msg: str) -> None:
    print(f"[tile] {msg}", flush=True)


def to_grayscale_u8(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 3:
        if arr.shape[0] == 1:
            arr = arr[0]
        elif arr.shape[0] >= 3:
            rgb = np.moveaxis(arr[:3], 0, -1).astype(np.float32)
            arr = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        else:
            arr = arr[0]
    arr = np.nan_to_num(arr.astype(np.float32))
    sample = arr[::4, ::4]
    valid = sample[np.isfinite(sample)]
    if valid.size == 0:
        return np.zeros(arr.shape, np.uint8)
    p_lo, p_hi = np.percentile(valid, [2.0, 98.0])
    if p_hi - p_lo < 1e-6:
        p_lo, p_hi = float(valid.min()), float(valid.max()) or 1.0
    stretched = np.clip((arr - p_lo) / max(p_hi - p_lo, 1e-6), 0, 1)
    return (stretched * 255).round().astype(np.uint8)


def head_size(url: str, timeout: int) -> int | None:
    try:
        r = requests.head(url, timeout=timeout, allow_redirects=True)
        r.raise_for_status()
        n = r.headers.get("Content-Length")
        return int(n) if n is not None else None
    except Exception as exc:  # noqa: BLE001
        log(f"HEAD failed ({exc}); will try to open directly")
        return None


def download(url: str, dest: Path, timeout: int, expected: int | None) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    log(f"downloading {urlparse(url).path.split('/')[-1]} "
        f"({(expected or 0) / 1e6:.0f} MB)")
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", expected or 0))
        with open(tmp, "wb") as fh, tqdm(total=total, unit="B", unit_scale=True,
                                         desc="download") as bar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    fh.write(chunk)
                    bar.update(len(chunk))
    tmp.replace(dest)
    return dest


def resolve_source(url: str, cfg: dict) -> tuple[str, str, int | None]:
    """Return (rasterio_path, mode, download_bytes)."""
    parsed = urlparse(url)
    if parsed.scheme in ("", "file"):
        p = Path(url)
        if p.exists():
            return str(p), "local", p.stat().st_size
        raise FileNotFoundError(url)

    raw_dir = ROOT / "data" / "raw"
    name = Path(parsed.path).name or "nac_source.tif"
    dest = raw_dir / name
    if dest.exists() and dest.stat().st_size > 1024:
        log(f"using cached {dest.name} ({dest.stat().st_size / 1e6:.0f} MB)")
        return str(dest), "download", dest.stat().st_size

    size = head_size(url, int(cfg["tiling"]["download_timeout_s"]))
    max_bytes = int(cfg["tiling"]["max_download_mb"]) * 1024 * 1024
    if size is not None and size <= max_bytes:
        return str(download(url, dest, int(cfg["tiling"]["download_timeout_s"]), size)), "download", size

    log(f"remote source is {((size or 0) / 1e6):.0f} MB; reading only the needed window")
    return "/vsicurl/" + url, "remote", size


def make_local_crs(lat: float, lon: float) -> CRS:
    return CRS.from_proj4(
        f"+proj=aeqd +lat_0={lat:.9f} +lon_0={lon:.9f} +R={MOON_RADIUS_M} "
        "+units=m +no_defs +type=crs"
    )


def source_center_lonlat(src: rasterio.io.DatasetReader) -> tuple[float, float]:
    bounds = src.bounds
    cx = (bounds.left + bounds.right) / 2.0
    cy = (bounds.bottom + bounds.top) / 2.0
    if src.crs is None:
        raise ValueError("source has no CRS; cannot georeference automatically")
    lon, lat = warp_transform(src.crs, GEO_CRS, [cx], [cy])
    return float(lat[0]), float(lon[0])


def read_source_window(src: rasterio.io.DatasetReader, cfg: dict,
                       extent_m: float) -> tuple[np.ndarray, object, str]:
    """Read roughly ``extent_m`` metres around the source centre."""
    t = src.transform
    if src.crs and src.crs.is_geographic:
        lat = (src.bounds.bottom + src.bounds.top) / 2.0
        m_per_deg_y = math.pi * MOON_RADIUS_M / 180.0
        m_per_deg_x = m_per_deg_y * max(0.2, math.cos(math.radians(lat)))
        gsd_x = abs(t.a) * m_per_deg_x
        gsd_y = abs(t.e) * m_per_deg_y
    else:
        gsd_x, gsd_y = abs(t.a), abs(t.e)
        # Heuristic: a CRS in feet, or a malformed transform.
        if gsd_x > 1000 or gsd_x == 0:
            gsd_x = 0.5
        if gsd_y > 1000 or gsd_y == 0:
            gsd_y = 0.5

    win_w = max(64, int(round(extent_m / gsd_x)))
    win_h = max(64, int(round(extent_m / gsd_y)))
    win_w = min(win_w, src.width)
    win_h = min(win_h, src.height)
    col_off = max(0, int(round((src.width - win_w) / 2.0)))
    row_off = max(0, int(round((src.height - win_h) / 2.0)))
    window = Window(col_off, row_off, win_w, win_h)

    log(f"reading source window {win_w}×{win_h} px at ({col_off},{row_off})")
    arr = src.read(window=window)
    gray = to_grayscale_u8(arr)
    win_transform = src.window_transform(window)
    return gray, win_transform, f"{win_w}x{win_h}"


def reproject_to_local(gray: np.ndarray, win_transform, src_crs,
                       cfg: dict) -> tuple[np.ndarray, rasterio.Affine, CRS, str]:
    """Warp the window into a local metric AEQD CRS at target GSD."""
    gsd = float(cfg["tiling"]["target_gsd"])
    tile_size = int(cfg["tiling"]["tile_size"])
    overlap = int(cfg["tiling"]["overlap"])
    max_tiles = int(cfg["tiling"]["max_tiles"])
    cols = max(1, math.ceil(math.sqrt(max_tiles)))
    grid_w = (cols - 1) * (tile_size - overlap) + tile_size
    extent_m = grid_w * gsd

    if src_crs is None:
        # No georeference: treat the pixels as a local metric frame.
        local_crs = make_local_crs(0.0, 0.0)
        h, w = gray.shape
        west = -w * gsd / 2.0
        north = h * gsd / 2.0
        transform = from_origin(west, north, gsd, gsd)
        return gray, transform, local_crs, "pixel frame"

    h, w = gray.shape
    left, top = win_transform.c, win_transform.f
    right = left + w * win_transform.a
    bottom = top + h * win_transform.e
    lat, lon = source_center_lonlat_from_bounds(src_crs, left, bottom, right, top)
    local_crs = make_local_crs(lat, lon)

    dst_bounds = transform_bounds(
        src_crs, local_crs, left, bottom, right, top, densify_pts=21
    )
    dst_w = max(1, int(round((dst_bounds[2] - dst_bounds[0]) / gsd)))
    dst_h = max(1, int(round((dst_bounds[3] - dst_bounds[1]) / gsd)))
    # Never allow an accidental giant output.
    scale = min(1.0, 6000.0 / max(dst_w, dst_h))
    if scale < 1.0:
        dst_w, dst_h = int(dst_w * scale), int(dst_h * scale)

    dst = np.zeros((dst_h, dst_w), np.uint8)
    dst_transform = from_origin(dst_bounds[0], dst_bounds[3], gsd, gsd)
    reproject(
        source=gray,
        destination=dst,
        src_transform=win_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=local_crs,
        resampling=Resampling.bilinear,
    )
    return dst, dst_transform, local_crs, f"AEQD {lat:.4f},{lon:.4f}"


def source_center_lonlat_from_bounds(src_crs, left, bottom, right, top):
    cx, cy = (left + right) / 2.0, (bottom + top) / 2.0
    lon, lat = warp_transform(src_crs, GEO_CRS, [cx], [cy])
    return float(lat[0]), float(lon[0])


def build_from_raster(url: str, cfg: dict) -> dict | None:
    tiling = cfg["tiling"]
    tile_size = int(tiling["tile_size"])
    overlap = int(tiling["overlap"])
    max_tiles = int(tiling["max_tiles"])
    gsd = float(tiling["target_gsd"])
    cols = max(1, math.ceil(math.sqrt(max_tiles)))
    rows = cols
    step = tile_size - overlap
    grid_w = (cols - 1) * step + tile_size
    grid_h = (rows - 1) * step + tile_size
    extent_m = max(grid_w, grid_h) * gsd

    path, mode, size = resolve_source(url, cfg)
    log(f"opening source ({mode})")
    try:
        src = rasterio.open(path)
    except Exception as exc:  # noqa: BLE001
        if mode == "remote" and cfg["tiling"].get("allow_full_download_for_remote"):
            log(f"remote open failed ({exc}); trying full download")
            raw_dir = ROOT / "data" / "raw"
            name = Path(urlparse(url).path).name or "nac_source.tif"
            path = str(download(url, raw_dir / name,
                                int(tiling["download_timeout_s"]), size))
            src = rasterio.open(path)
            mode = "download"
        else:
            raise

    with src:
        log(f"source: {src.width}×{src.height} px, {src.count} band(s), CRS={src.crs}")
        gray, win_transform, _ = read_source_window(src, cfg, extent_m)
        mosaic, mosaic_transform, local_crs, crs_note = reproject_to_local(
            gray, win_transform, src.crs, cfg
        )
        upper_url = url.upper()
        src_type = "real_nac" if any(k in upper_url for k in
                                     ("LROC", "NAC", "ORTHOMOSAIC", "APOLLO17")) else "real_geotiff"

    # Crop to the exact tile grid so viewer bounds and hazard cells line up.
    if mosaic.shape[1] < grid_w or mosaic.shape[0] < grid_h:
        cols = max(1, min(cols, (mosaic.shape[1] - tile_size) // step + 1))
        rows = max(1, min(rows, (mosaic.shape[0] - tile_size) // step + 1))
        grid_w = (cols - 1) * step + tile_size
        grid_h = (rows - 1) * step + tile_size
    x_off = (mosaic.shape[1] - grid_w) // 2
    y_off = (mosaic.shape[0] - grid_h) // 2
    mosaic = mosaic[y_off:y_off + grid_h, x_off:x_off + grid_w]
    west = mosaic_transform.c + x_off * gsd
    north = mosaic_transform.f - y_off * gsd
    south = north - grid_h * gsd
    east = west + grid_w * gsd
    transform = from_origin(west, north, gsd, gsd)

    if src_type == "real_nac":
        site_name = "Apollo 17 — Taurus-Littrow (LROC NAC)"
        credit = "NASA/GSFC/ASU LROC; USGS Astrogeology (Apollo 17 NAC orthomosaic)"
    else:
        site_name = "LROC NAC site"
        credit = "NASA/GSFC/ASU LROC"

    return finalise_tiles(
        mosaic, transform, local_crs, cfg, cols, rows, step, tile_size, gsd,
        source={
            "type": src_type,
            "url": url,
            "mode": mode,
            "product": "Moon Apollo 17 LROC NAC Landing Site Orthomosaic 50cm"
            if "APOLLO17" in url.upper() else urlparse(url).path.split("/")[-1],
            "credit": credit,
            "crs_note": crs_note,
        },
        site_name=site_name,
    )


def build_synthetic(cfg: dict, reason: str = "") -> dict:
    tiling = cfg["tiling"]
    tile_size = int(tiling["tile_size"])
    overlap = int(tiling["overlap"])
    max_tiles = int(tiling["max_tiles"])
    gsd = float(tiling["target_gsd"])
    cols = max(1, math.ceil(math.sqrt(max_tiles)))
    rows = cols
    step = tile_size - overlap
    grid_w = (cols - 1) * step + tile_size

    site = cfg["project"]["site"]
    lat, lon = float(site["center_lat"]), float(site["center_lon"])
    local_crs = make_local_crs(lat, lon)

    log(f"generating synthetic fallback terrain ({reason})")
    rng = np.random.default_rng(int(cfg["project"].get("seed", 42)) + 99)
    scene = render_scene(
        grid_w,
        rng,
        n_boulders=110,
        boulder_size_px=(3, 42),
        n_craters=20,
        n_micro_rocks=420,
    )
    mosaic = to_uint8(scene.image)
    west = -grid_w * gsd / 2.0
    north = grid_w * gsd / 2.0
    transform = from_origin(west, north, gsd, gsd)

    return finalise_tiles(
        mosaic, transform, local_crs, cfg, cols, rows, step, tile_size, gsd,
        source={
            "type": "synthetic_fallback",
            "url": "",
            "mode": "generated",
            "product": "Procedural lunar regolith (no network data)",
            "credit": "Synthetic renderer",
            "crs_note": reason or "network unavailable",
        },
        site_name=str(site.get("name", "Synthetic lunar site")),
    )


def finalise_tiles(mosaic: np.ndarray, transform, local_crs: CRS, cfg: dict,
                   cols: int, rows: int, step: int, tile_size: int, gsd: float,
                   *, source: dict, site_name: str) -> dict:
    img_dir = ROOT / "data" / "tiles" / "images"
    lbl_dir = ROOT / "data" / "tiles" / "labels"
    meta_dir = ROOT / "data" / "labels"
    viewer_dir = ROOT / "viewer" / "data"
    for d in (img_dir, lbl_dir, meta_dir, viewer_dir):
        d.mkdir(parents=True, exist_ok=True)

    west, north = transform.c, transform.f
    height, width = mosaic.shape
    south = north - height * gsd
    east = west + width * gsd

    def to_geo(x, y):
        lon, lat = warp_transform(local_crs, GEO_CRS, [x], [y])
        return lon[0], lat[0]

    tiles_meta: list[dict] = []
    for row in range(rows):
        for col in range(cols):
            y = row * step
            x = col * step
            if y + tile_size > height or x + tile_size > width:
                continue
            chip = mosaic[y:y + tile_size, x:x + tile_size]
            name = f"nac_{row:02d}_{col:02d}"
            tile_transform = from_origin(
                west + x * gsd,
                north - y * gsd,
                gsd,
                gsd,
            )
            tif_path = img_dir / f"{name}.tif"
            with rasterio.open(
                tif_path, "w", driver="GTiff",
                height=tile_size, width=tile_size, count=1, dtype="uint8",
                crs=local_crs, transform=tile_transform, compress="deflate",
            ) as dst:
                dst.write(chip, 1)
            cv2.imwrite(str(img_dir / f"{name}.png"), chip)
            # Real tiles have no manual labels; an empty file is explicit.
            (lbl_dir / f"{name}.txt").write_text("", encoding="utf-8")

            cx = tile_transform.c + tile_size * gsd / 2.0
            cy = tile_transform.f - tile_size * gsd / 2.0
            t_lon, t_lat = to_geo(cx, cy)
            tiles_meta.append({
                "name": name,
                "tif": str(tif_path.relative_to(ROOT)).replace("\\", "/"),
                "png": str((img_dir / f"{name}.png").relative_to(ROOT)).replace("\\", "/"),
                "row": row,
                "col": col,
                "gsd_m": gsd,
                "bounds_m": [round(cx - tile_size * gsd / 2.0, 3),
                             round(cy - tile_size * gsd / 2.0, 3),
                             round(cx + tile_size * gsd / 2.0, 3),
                             round(cy + tile_size * gsd / 2.0, 3)],
                "center_m": [round(cx, 3), round(cy, 3)],
                "center_lat": round(float(t_lat), 7),
                "center_lon": round(float(t_lon), 7),
            })

    # Basemap for the browser (image overlay in CRS.Simple metres).
    view_w = min(2048, width)
    view_h = round(height * view_w / width)
    mosaic_small = cv2.resize(mosaic, (view_w, view_h), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(viewer_dir / "mosaic.jpg"), mosaic_small,
                [cv2.IMWRITE_JPEG_QUALITY, 90])

    center_lon, center_lat = to_geo(
        (west + east) / 2.0, (south + north) / 2.0
    )
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "site": {
            "name": site_name,
            "center_lat": round(float(center_lat), 7),
            "center_lon": round(float(center_lon), 7),
            "radius_m": MOON_RADIUS_M,
            "map_frame": "local lunar Azimuthal Equidistant, metres",
        },
        "source": source,
        "crs": local_crs.to_proj4(),
        "gsd_m": gsd,
        "tile_size": tile_size,
        "overlap_px": tile_size - step,
        "step_px": step,
        "cols": cols,
        "rows": rows,
        "tile_count": len(tiles_meta),
        "mosaic": {
            "path": "viewer/data/mosaic.jpg",
            "bounds_m": [round(west, 3), round(south, 3), round(east, 3), round(north, 3)],
            "width_px": width,
            "height_px": height,
        },
        "tiles": tiles_meta,
    }
    with open(meta_dir / "tiles_metadata.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
    log(f"wrote {len(tiles_meta)} tiles + mosaic ({width}×{height} px)")
    return meta


def main() -> None:
    cfg = load_config()
    t0 = time.time()
    errors: list[str] = []
    meta = None
    for url in cfg["tiling"].get("nac_urls") or []:
        try:
            meta = build_from_raster(url, cfg)
            break
        except Exception as exc:  # noqa: BLE001
            log(f"source failed: {exc}")
            errors.append(f"{url}: {exc}")

    if meta is None:
        if not cfg["tiling"].get("allow_synthetic_fallback", True):
            raise SystemExit("No NAC source available and synthetic fallback is disabled. "
                             + " | ".join(errors))
        meta = build_synthetic(cfg, reason="; ".join(errors)[:500])

    print(f"\n✓ Tiles ready in {time.time() - t0:.1f}s")
    print(f"  site:  {meta['site']['name']} "
          f"({meta['site']['center_lat']:.4f}°, {meta['site']['center_lon']:.4f}°)")
    print(f"  GSD:   {meta['gsd_m']} m/px")
    print(f"  tiles: {meta['tile_count']} ({meta['cols']}×{meta['rows']})")
    print(f"  source: {meta['source']['type']} — {meta['source']['product']}")


if __name__ == "__main__":
    main()
