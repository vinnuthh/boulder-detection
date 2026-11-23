#!/usr/bin/env python
"""Stage 4 — detect boulders, estimate metric size, build hazard grids.

Pipeline
--------
512px NAC tiles → YOLOv8n → deduplicated detections → GSD size estimate
→ 50m hazard cells for lander + rover profiles → GeoJSON / GeoTIFF / PNG /
hazard_meta.json for the Leaflet viewer.

If no trained checkpoint is available (or the model finds nothing), a
classical bright-blob + cast-shadow detector keeps the demo alive.  Every
detection records exactly where its size came from.

Usage:
    python scripts/03_infer_and_hazard.py --weights models/weights/best.pt --conf 0.25
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import rasterio
try:  # the demo venv ships PyYAML only
    from ruamel.yaml import YAML
    yaml = YAML(typ="safe")
except ImportError:  # pragma: no cover - fallback for minimal environments
    import yaml as _pyyaml

    class _SafeLoader:
        @staticmethod
        def load(fh):
            return _pyyaml.safe_load(fh)

    yaml = _SafeLoader()
from rasterio.crs import CRS
from rasterio.transform import from_origin
from rasterio.warp import transform as warp_transform
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
GEO_CRS = CRS.from_proj4("+proj=longlat +R=1737400 +no_defs")


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.load(fh)


def log(msg: str) -> None:
    print(f"[infer] {msg}", flush=True)


# ---------------------------------------------------------------------------
# classical fallback detector
# ---------------------------------------------------------------------------
def classical_detect(png_path: Path, gsd: float) -> list[dict]:
    """Multi-scale DoG detector, shared with the upload API.

    Delegating to ``safety_model._classical_detections`` keeps the site map and
    the upload preview consistent: boxes cover the whole silhouette (not just
    the sunlit crown), so diameters match the 1.5-20 m boulder range of the
    source tiles.  Returns boxes in tile pixels.
    """
    bgr = cv2.imread(str(png_path), cv2.IMREAD_COLOR)
    if bgr is None:
        return []
    from safety_model import _classical_detections  # local import to avoid a cycle

    dets = _classical_detections(bgr, gsd, _lander_thresholds())
    return [
        {
            "xyxy": [float(d["x"]), float(d["y"]), float(d["x"] + d["w"]), float(d["y"] + d["h"])],
            "conf": float(d["confidence"]),
            "source": "classical",
        }
        for d in dets
    ]


_TH_CACHE: dict | None = None


def _lander_thresholds() -> dict:
    global _TH_CACHE
    if _TH_CACHE is None:
        _TH_CACHE = load_yaml(ROOT / "thresholds.yaml")["profiles"]["lander"]
    return _TH_CACHE


# ---------------------------------------------------------------------------
# YOLO inference
# ---------------------------------------------------------------------------
def yolo_detect(weights: Path, tiles: list[dict], cfg: dict, conf: float,
                device: str) -> tuple[list[dict], str]:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    detections: list[dict] = []
    half = bool(cfg["model"].get("half_precision", True)) and device != "cpu"
    imgsz = int(cfg["model"].get("imgsz", 512))
    max_det = int(cfg["inference"].get("max_det", 300))
    for tile in tqdm(tiles, desc="YOLOv8n inference", unit="tile"):
        png = ROOT / tile["png"]
        res = model.predict(
            source=str(png), conf=conf, iou=float(cfg["inference"].get("iou", 0.5)),
            imgsz=imgsz, device=device, half=half, max_det=max_det,
            verbose=False,
        )[0]
        if res.boxes is None or len(res.boxes) == 0:
            continue
        xyxy = res.boxes.xyxy.detach().cpu().numpy()
        confs = res.boxes.conf.detach().cpu().numpy()
        for box, c in zip(xyxy, confs):
            detections.append({
                "xyxy": [float(v) for v in box],
                "conf": float(c),
                "source": "yolov8n",
                "tile": tile["name"],
            })
    return detections, "yolov8n"


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------
def iou_xyxy(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, x2 - x1), max(0.0, y2 - y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-9)


def nms(dets: list[dict], thr: float = 0.35) -> list[dict]:
    dets = sorted(dets, key=lambda d: d["conf"], reverse=True)
    kept: list[dict] = []
    for d in dets:
        if any(iou_xyxy(d["bbox_m"], k["bbox_m"]) > thr for k in kept):
            continue
        kept.append(d)
    return kept


def world_detections(tile_boxes: list[dict], tile_meta: dict, cfg: dict,
                     local_crs) -> list[dict]:
    tile_name = tile_meta["name"]
    tif = ROOT / tile_meta["tif"]
    out: list[dict] = []
    with rasterio.open(tif) as src:
        transform = src.transform
        gsd = abs(transform.a)  # metres / pixel — native source resolution
        _ = src.crs
    for d in tile_boxes:
        x1p, y1p, x2p, y2p = d["xyxy"]
        x1p = float(np.clip(x1p, 0, 512))
        y1p = float(np.clip(y1p, 0, 512))
        x2p = float(np.clip(x2p, 0, 512))
        y2p = float(np.clip(y2p, 0, 512))
        if x2p <= x1p or y2p <= y1p:
            continue
        ax, ay = transform * (x1p, y1p)
        bx, by = transform * (x2p, y2p)
        xmin, xmax = min(ax, bx), max(ax, bx)
        ymin, ymax = min(ay, by), max(ay, by)
        diameter_px = max(x2p - x1p, y2p - y1p)
        diameter_m = diameter_px * gsd
        cx, cy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
        lon, lat = warp_transform(local_crs, GEO_CRS, [cx], [cy])
        out.append({
            "x": round(cx, 3),
            "y": round(cy, 3),
            "lon": round(float(lon[0]), 7),
            "lat": round(float(lat[0]), 7),
            "bbox_m": [round(xmin, 3), round(ymin, 3), round(xmax, 3), round(ymax, 3)],
            "conf": round(float(d["conf"]), 4),
            "diameter_px": round(float(diameter_px), 2),
            "diameter_m": round(float(diameter_m), 3),
            "gsd_m": round(float(gsd), 4),
            "tile": tile_name,
            "source": d["source"],
        })
    return out


# ---------------------------------------------------------------------------
# hazard grid
# ---------------------------------------------------------------------------
def classify_size(diameter_m: float, th: dict) -> int:
    if diameter_m >= float(th["size"]["unsafe_min_diameter_m"]):
        return 2
    if diameter_m >= float(th["size"]["safe_max_diameter_m"]):
        return 1
    return 0


def classify_density(count: int, th: dict) -> int:
    if count >= int(th["density"]["unsafe_min_count"]):
        return 2
    if count > int(th["density"]["safe_max_count"]):
        return 1
    return 0


def build_hazard_grid(detections: list[dict], profiles: dict, colors: dict,
                      bounds: list[float], cell_m: float, min_conf: float,
                      local_crs, viewer_dir: Path, default_profile: str) -> dict:
    west, south, east, north = bounds
    cols = max(1, int(math.ceil((east - west) / cell_m)))
    rows = max(1, int(math.ceil((north - south) / cell_m)))
    # Anchor the grid to the mosaic's north-west corner so the row/column
    # mapping is identical in Python and in the viewer.  The grid can overhang
    # the mosaic's south/east edge by less than one cell; those cells are
    # simply empty (safe).
    grid_bounds = [west, north - rows * cell_m, west + cols * cell_m, north]
    log(f"hazard grid {cols}×{rows} cells @ {cell_m} m")

    active = [d for d in detections if d["conf"] >= min_conf]
    cell_data: dict[tuple[int, int], dict] = {}
    for d in active:
        col = int(math.floor((d["x"] - west) / cell_m))
        row = int(math.floor((north - d["y"]) / cell_m))
        if not (0 <= col < cols and 0 <= row < rows):
            continue
        c = cell_data.setdefault((row, col), {"count": 0, "max_d": 0.0})
        c["count"] += 1
        c["max_d"] = max(c["max_d"], d["diameter_m"])

    out_profiles: dict[str, dict] = {}
    for pname, th in profiles.items():
        zones = np.zeros((rows, cols), np.uint8)
        for (row, col), c in cell_data.items():
            z = max(classify_size(c["max_d"], th), classify_density(c["count"], th))
            zones[row, col] = z
        stats = {
            "safe_cells": int((zones == 0).sum()),
            "caution_cells": int((zones == 1).sum()),
            "unsafe_cells": int((zones == 2).sum()),
            "total_cells": int(rows * cols),
            "cells_with_detections": int(len(cell_data)),
            "detections_used": len(active),
            "max_diameter_m": round(max((d["diameter_m"] for d in active), default=0.0), 3),
        }

        # RGBA PNG (upscaled nearest-neighbour for crisp 50m blocks).
        palette = {
            0: _hex_rgb(colors["safe"]),
            1: _hex_rgb(colors["caution"]),
            2: _hex_rgb(colors["unsafe"]),
        }
        rgba = np.zeros((rows, cols, 4), np.uint8)
        for z, rgb in palette.items():
            rgba[zones == z] = (*rgb, 235)
        png = cv2.resize(rgba, (cols * 6, rows * 6), interpolation=cv2.INTER_NEAREST)
        png_path = viewer_dir / f"hazard_{pname}.png"
        cv2.imwrite(str(png_path), cv2.cvtColor(png, cv2.COLOR_RGBA2BGRA))

        tif_path = viewer_dir / f"hazard_{pname}.tif"
        with rasterio.open(
            tif_path, "w", driver="GTiff", height=rows, width=cols, count=1,
            dtype="uint8", crs=local_crs,
            transform=from_origin(west, grid_bounds[3], cell_m, cell_m),
            nodata=255, compress="lzw",
        ) as dst:
            dst.write(zones, 1)
            dst.update_tags(
                profile=pname, cell_size_m=cell_m,
                legend="0=safe,1=caution,2=unsafe",
                thresholds=json.dumps(th),
            )
        if pname == default_profile:
            shutil.copy2(tif_path, viewer_dir / "hazard.tif")

        out_profiles[pname] = {
            "label": th.get("label", pname.title()),
            "description": th.get("description", ""),
            "thresholds": {
                "size": th["size"],
                "density": th["density"],
            },
            "stats": stats,
            "hazard_png": f"data/hazard_{pname}.png",
            "hazard_tif": f"data/hazard_{pname}.tif",
            "zones": zones.tolist(),
        }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "grid": {
            "cell_size_m": cell_m,
            "cols": cols,
            "rows": rows,
            "bounds_m": [round(v, 3) for v in grid_bounds],
            "legend": {"0": "safe", "1": "caution", "2": "unsafe"},
        },
        "colors": colors,
        "min_confidence": min_conf,
        "profiles": out_profiles,
        "default_profile": default_profile,
        "detections_total": len(detections),
        "detections_above_min_conf": len(active),
    }


def _hex_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="models/weights/best.pt")
    parser.add_argument("--conf", type=float, default=None,
                        help="minimum confidence used for hazard stats")
    parser.add_argument("--profile", default=None,
                        help="default hazard profile written to hazard.tif")
    parser.add_argument("--detector", choices=["auto", "yolo", "classical"], default="auto")
    parser.add_argument("--device", default=None)
    parser.add_argument("--min-export-conf", type=float, default=0.05)
    args = parser.parse_args()

    cfg = load_yaml(ROOT / "config.yaml")
    thresholds = load_yaml(ROOT / "thresholds.yaml")
    meta_path = ROOT / "data" / "labels" / "tiles_metadata.json"
    if not meta_path.exists():
        raise SystemExit("No tiles metadata. Run: python scripts/01_fetch_and_tile.py")
    tiles_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    tiles = tiles_meta["tiles"]
    if not tiles:
        raise SystemExit("No tiles found. Run: python scripts/01_fetch_and_tile.py")

    conf = float(args.conf if args.conf is not None else cfg["inference"]["conf"])
    default_profile = args.profile or cfg["hazard"].get("default_profile", "lander")
    if default_profile not in thresholds["profiles"]:
        raise SystemExit(f"Unknown profile '{default_profile}'")
    viewer_dir = ROOT / "viewer" / "data"
    viewer_dir.mkdir(parents=True, exist_ok=True)

    weights = Path(args.weights)
    if not weights.is_absolute():
        weights = ROOT / weights
    device = args.device
    if device is None:
        try:
            import torch
            device = "0" if torch.cuda.is_available() else "cpu"
        except Exception:  # noqa: BLE001
            device = "cpu"

    detector_used = "classical"
    raw_dets: list[dict] = []
    t0 = time.time()
    if args.detector in ("auto", "yolo") and weights.exists():
        try:
            raw_dets, detector_used = yolo_detect(
                weights, tiles, cfg, args.min_export_conf, device
            )
            log(f"YOLO exported {len(raw_dets)} raw boxes")
        except Exception as exc:  # noqa: BLE001
            log(f"YOLO inference failed: {exc}")
            if args.detector == "yolo":
                raise
    elif args.detector == "auto":
        log(f"no weights at {weights}; falling back to classical CV")

    if args.detector == "classical" or len(raw_dets) < 3:
        if not cfg["inference"].get("allow_classical_fallback", True):
            raise SystemExit("No detections and classical fallback disabled")
        log("supplementing with classical shape detector")
        classical_total = 0
        for tile in tqdm(tiles, desc="classical detection", unit="tile"):
            for d in classical_detect(ROOT / tile["png"], tiles_meta["gsd_m"]):
                d["tile"] = tile["name"]
                raw_dets.append(d)
                classical_total += 1
        detector_used = "yolov8n+classical" if detector_used == "yolov8n" else "classical"
        log(f"classical exported {classical_total} raw boxes")

    by_tile: dict[str, list[dict]] = {}
    for d in raw_dets:
        by_tile.setdefault(d["tile"], []).append(d)

    local_crs = CRS.from_proj4(tiles_meta["crs"])
    detections: list[dict] = []
    for tile in tiles:
        boxes = by_tile.get(tile["name"], [])
        if not boxes:
            continue
        detections.extend(world_detections(boxes, tile, cfg, local_crs))

    detections = [d for d in detections if d["conf"] >= args.min_export_conf]
    detections = nms(detections, thr=0.35)
    detections.sort(key=lambda d: d["conf"], reverse=True)
    for i, d in enumerate(detections):
        d["id"] = f"boulder_{i:04d}"
    log(f"{len(detections)} unique boulders after NMS")

    # ---------------------------------------------------------------- GeoJSON
    features = []
    for d in detections:
        features.append({
            "type": "Feature",
            "id": d["id"],
            "geometry": {"type": "Point", "coordinates": [d["x"], d["y"]]},
            "properties": {
                "id": d["id"],
                "class": "boulder",
                "confidence": d["conf"],
                "diameter_m": d["diameter_m"],
                "diameter_px": d["diameter_px"],
                "gsd_m": d["gsd_m"],
                "tile": d["tile"],
                "source": d["source"],
                "lat": d["lat"],
                "lon": d["lon"],
                "bbox_m": d["bbox_m"],
            },
        })
    geojson = {
        "type": "FeatureCollection",
        "name": "lunar_boulder_detections",
        "crs": {"type": "name", "properties": {
            "name": "local lunar Azimuthal Equidistant (metres)"}},
        "metadata": {
            "site": tiles_meta["site"],
            "gsd_m": tiles_meta["gsd_m"],
            "detector": detector_used,
            "min_export_conf": args.min_export_conf,
            "count": len(detections),
        },
        "features": features,
    }
    with open(viewer_dir / "detections.geojson", "w", encoding="utf-8") as fh:
        json.dump(geojson, fh, separators=(",", ":"))

    # --------------------------------------------------------------- hazard
    hazard = build_hazard_grid(
        detections,
        thresholds["profiles"],
        thresholds["colors"],
        tiles_meta["mosaic"]["bounds_m"],
        float(cfg["hazard"]["grid_cell_size_m"]),
        conf,
        local_crs,
        viewer_dir,
        default_profile,
    )
    hazard["site"] = tiles_meta["site"]
    hazard["gsd_m"] = tiles_meta["gsd_m"]
    hazard["basemap"] = {
        "path": "data/mosaic.jpg",
        "bounds_m": tiles_meta["mosaic"]["bounds_m"],
        "width_px": tiles_meta["mosaic"]["width_px"],
        "height_px": tiles_meta["mosaic"]["height_px"],
        "gsd_m": tiles_meta["gsd_m"],
        "source": tiles_meta["source"],
    }
    hazard["viewer"] = cfg.get("viewer", {})
    hazard["detector"] = detector_used
    hazard["weights"] = str(weights) if weights.exists() else None
    report_path = ROOT / "models" / "weights" / "training_report.json"
    if report_path.exists():
        try:
            hazard["training"] = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            pass
    with open(viewer_dir / "hazard_meta.json", "w", encoding="utf-8") as fh:
        json.dump(hazard, fh, indent=1)

    # Keep a copy of detections next to the hazard products for auditability.
    with open(ROOT / "data" / "labels" / "detections.geojson", "w", encoding="utf-8") as fh:
        json.dump(geojson, fh, separators=(",", ":"))

    elapsed = time.time() - t0
    print(f"\n✓ Inference + hazard complete in {elapsed:.1f}s")
    print(f"  detector: {detector_used}")
    print(f"  boulders: {len(detections)} (>= {conf:.2f} for hazard stats: "
          f"{hazard['detections_above_min_conf']})")
    for pname, p in hazard["profiles"].items():
        s = p["stats"]
        print(f"  {pname:6s}: {s['safe_cells']} safe / {s['caution_cells']} caution / "
              f"{s['unsafe_cells']} unsafe cells")
    print(f"  viewer:   {viewer_dir / 'detections.geojson'}")
    print(f"  hazard:   {viewer_dir / 'hazard_meta.json'}")


if __name__ == "__main__":
    main()
