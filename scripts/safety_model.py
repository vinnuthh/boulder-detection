"""Landing-safety model shared by training and the upload API.

PyTorch is blocked by Windows Smart App Control on the demo machine, so this
module trains a compact classical-vision model instead of pretending a YOLO
checkpoint exists.  The model classifies a 50 m x 50 m lunar surface patch as
SAFE / CAUTION / UNSAFE for a lander, using HOG texture features plus simple
photometric statistics.  Boulder diameters and counts for the training labels
come from the exact synthetic YOLO annotations (0.5 m/px).
"""
from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
try:  # ruamel keeps YAML anchors/ordering; PyYAML is a fine fallback
    from ruamel.yaml import YAML
    _YAML = YAML(typ="safe")
except ModuleNotFoundError:  # pragma: no cover - depends on the interpreter
    import yaml as _pyyaml
    _YAML = None

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "weights" / "landing_safety.joblib"
REPORT_PATH = ROOT / "models" / "weights" / "safety_report.json"

FEATURE_VERSION = 3
PATCH_PX = 128
CELL_M = 50.0
TRAIN_GSD = 0.5
CELL_PX = int(round(CELL_M / TRAIN_GSD))  # 100 px
CLASS_NAMES = ("safe", "caution", "unsafe")

_HOG = cv2.HOGDescriptor(
    (PATCH_PX, PATCH_PX),
    (32, 32),
    (16, 16),
    (16, 16),
    9,
)


def load_thresholds(path: Path | None = None) -> dict[str, Any]:
    p = path or (ROOT / "thresholds.yaml")
    with open(p, encoding="utf-8") as fh:
        if _YAML is not None:
            return _YAML.load(fh)
        return _pyyaml.safe_load(fh)


def zone_from_values(diameter_m: float, count: int, th: dict[str, Any]) -> int:
    """0 safe, 1 caution, 2 unsafe for one 50 m cell."""
    size = th["size"]
    density = th["density"]
    if diameter_m >= float(size["unsafe_min_diameter_m"]):
        return 2
    if count >= int(density["unsafe_min_count"]):
        return 2
    if diameter_m >= float(size["safe_max_diameter_m"]):
        return 1
    if count > int(density["safe_max_count"]):
        return 1
    return 0


def extract_features(bgr: np.ndarray) -> np.ndarray:
    """Deterministic feature vector for a 128 x 128 BGR patch."""
    if bgr.ndim == 2:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    img = cv2.resize(bgr, (PATCH_PX, PATCH_PX), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hog = _HOG.compute(gray).ravel()

    hist = cv2.calcHist([gray], [0], None, [32], [0, 256]).ravel()
    hist = hist / max(1.0, float(gray.size))

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    ghist = np.histogram(mag, bins=16, range=(0.0, 512.0))[0].astype(np.float32)
    ghist = ghist / max(1.0, float(gray.size))

    g = gray.astype(np.float32) / 255.0
    p10, p50, p90, p99 = np.percentile(g, [10, 50, 90, 99])
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    edges = cv2.Canny(gray, 40, 120)
    bright = float((gray >= 190).mean())
    dark = float((gray <= 55).mean())
    stats = np.array(
        [
            float(g.mean()),
            float(g.std()),
            float(p10),
            float(p50),
            float(p90),
            float(p99),
            float(edges.mean() / 255.0),
            bright,
            dark,
            float(lap.var() / 1000.0),
            float(np.abs(lap).mean() / 255.0),
        ],
        dtype=np.float32,
    )

    # Bright-body blob statistics at two thresholds: these capture boulder
    # size/count directly and separate 1-3 m rocks from both empty terrain and
    # large lethal blocks.
    blob_feats: list[float] = []
    for thresh in (140, 175):
        mask = (gray >= thresh).astype(np.uint8)
        n, _, cstats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        widths: list[int] = []
        heights: list[int] = []
        areas: list[int] = []
        for i in range(1, n):
            area = int(cstats[i, cv2.CC_STAT_AREA])
            bw = int(cstats[i, cv2.CC_STAT_WIDTH])
            bh = int(cstats[i, cv2.CC_STAT_HEIGHT])
            if area >= 4 and bw >= 2 and bh >= 2:
                areas.append(area)
                widths.append(bw)
                heights.append(bh)
        max_dim = [max(w, h) for w, h in zip(widths, heights)]
        blob_feats.extend(
            [
                float(len(areas)),
                float(np.max(max_dim)) if max_dim else 0.0,
                float(np.percentile(max_dim, 90)) if max_dim else 0.0,
                float(np.mean(max_dim)) if max_dim else 0.0,
                float(sum(1 for d in max_dim if d >= 3)),
                float(sum(1 for d in max_dim if d >= 6)),
                float(sum(1 for d in max_dim if d >= 10)),
                float(sum(1 for d in max_dim if d >= 16)),
                float(sum(areas) / gray.size),
                float(mask.mean()),
            ]
        )
    blob_feats = np.asarray(blob_feats, dtype=np.float32)

    small = cv2.resize(g, (16, 16), interpolation=cv2.INTER_AREA).ravel()
    return np.concatenate([hog, hist, ghist, stats, blob_feats, small]).astype(np.float32)


def padded_crop(bgr: np.ndarray, cx: float, cy: float, size: int = PATCH_PX) -> np.ndarray:
    half = size // 2
    pad = half + 2
    padded = cv2.copyMakeBorder(bgr, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
    x = int(round(cx)) + pad
    y = int(round(cy)) + pad
    return padded[y - half : y + half, x - half : x + half]


def cell_labels_from_yolo(
    label_path: Path, image_px: int, gsd: float, th: dict[str, Any], cells: int = 5
) -> list[int]:
    """Return per-cell zones from a YOLO label file (row-major)."""
    grid_d = np.zeros((cells, cells), dtype=np.float32)
    grid_n = np.zeros((cells, cells), dtype=np.int32)
    lines = label_path.read_text(encoding="utf-8").strip().splitlines()
    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue
        _, cx, cy, w, h = parts[:5]
        # Synthetic label rows are 5x5 cells of the source tile.  A boulder is
        # assigned to the cell containing its centre.
        col = min(cells - 1, max(0, int(float(cx) * cells)))
        row = min(cells - 1, max(0, int(float(cy) * cells)))
        grid_n[row, col] += 1
        grid_d[row, col] = max(grid_d[row, col], max(float(w), float(h)) * image_px * gsd)
    return [
        zone_from_values(float(grid_d[r, c]), int(grid_n[r, c]), th)
        for r in range(cells)
        for c in range(cells)
    ]


def decode_data_url(data_url: str) -> np.ndarray:
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    raw = base64.b64decode(data_url)
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    return img


def _classical_detections(
    bgr: np.ndarray, gsd: float, th: dict[str, Any], min_conf: float = 0.3
) -> list[dict[str, Any]]:
    """Multi-scale difference-of-Gaussians detector for upload overlays.

    A single DoG threshold only catches the sunlit crown of each rock, so the
    old boxes were small and most boulders were missed.  Running three scales
    (large, medium, small rocks), filling each contour, expanding the box a
    little and shifting it down-right (towards the cast shadow) recovers the
    whole silhouette.  The trained patch classifier makes the safety decision;
    these boxes explain which rocks drove it and are labelled as classical.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g = clahe.apply(gray).astype(np.float32)
    h, w = gray.shape

    # (percentile, Gaussian sigma, closing kernel, box expansion, down-right shift)
    scales = (
        (99.0, 9.0, 7, 1.2, 0.1),
        (98.0, 6.0, 5, 1.3, 0.1),
        (98.0, 3.0, 4, 1.25, 0.1),
    )
    boxes: list[tuple[int, int, int, int]] = []
    scores: list[float] = []
    boxes_out: list[dict[str, Any]] = []

    for pct, sigma, close_k, expand, shift in scales:
        dog = g - cv2.GaussianBlur(g, (0, 0), sigma)
        thresh = max(4.0, float(np.percentile(dog, pct)))
        mask = (dog >= thresh).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((close_k, close_k), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            cx, cy, cw, ch = cv2.boundingRect(cnt)
            if cw < 3 or ch < 3 or cw > 90 or ch > 90:
                continue
            area = cv2.contourArea(cnt)
            if area < 8 or area > 5000:
                continue
            aspect = max(cw, ch) / max(1.0, min(cw, ch))
            if aspect > 3.2:
                continue
            fill = area / max(1.0, cw * ch)
            if fill < 0.15:
                continue

            bw, bh = cw * expand, ch * expand
            bx = cx + cw / 2 - bw / 2 + shift * cw
            by = cy + ch / 2 - bh / 2 + shift * ch
            x0, y0 = max(0, int(round(bx))), max(0, int(round(by)))
            x1, y1 = min(w, int(round(bx + bw))), min(h, int(round(by + bh)))
            if x1 - x0 < 4 or y1 - y0 < 4:
                continue
            if max(x1 - x0, y1 - y0) > 48:
                # A box this large is several rocks or a crater merged together.
                continue

            # A boulder is brighter than the shadow it casts to the lower-right.
            tw, thh = max(2, (x1 - x0) // 2), max(2, (y1 - y0) // 2)
            lit = float(g[y0 : y0 + thh, x0 : x0 + tw].mean())
            sx, sy = min(w - tw, x0 + tw), min(h - thh, y0 + thh)
            shadow = float(g[sy : sy + thh, sx : sx + tw].mean()) if tw and thh else lit
            contrast = lit - shadow
            # Smooth, non-saturating contrast-to-confidence map: bright rocks
            # with a strong cast shadow score highest, but a linear clip would
            # pin almost every kept box at the ceiling and look fabricated.
            conf = float(np.clip(0.10 + 0.85 * contrast / (contrast + 35.0), 0.05, 0.95))
            if conf < min_conf:
                continue

            diameter_m = max(x1 - x0, y1 - y0) * gsd
            boxes.append((x0, y0, x1 - x0, y1 - y0))
            scores.append(conf)
            boxes_out.append(
                {
                    "x": int(x0),
                    "y": int(y0),
                    "w": int(x1 - x0),
                    "h": int(y1 - y0),
                    "diameter_m": round(float(diameter_m), 2),
                    "confidence": round(conf, 3),
                    "zone": zone_from_values(diameter_m, 0, th),
                }
            )

    if not boxes:
        return []
    keep = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.05, nms_threshold=0.3)
    idx = [int(i) for i in np.array(keep).flatten()]

    # A 512 px tile holds roughly 5-25 real boulders, but the three scales
    # over-fire on texture and crater rims.  Keep the highest-contrast boxes so
    # the overlay stays readable (and precision stays usable), scaling the
    # budget with image area so larger inputs still show their extra rocks.
    budget = max(12, int(round(15 * (w * h) / (512 * 512))))
    if len(idx) > budget:
        idx = sorted(idx, key=lambda i: scores[i], reverse=True)[:budget]

    out = [boxes_out[i] for i in idx]
    out.sort(key=lambda d: d["diameter_m"], reverse=True)
    return out[:400]


def _largest_safe_component(zones: np.ndarray) -> list[tuple[int, int]]:
    """4-connected component of safe cells with the largest area."""
    rows, cols = zones.shape
    seen = np.zeros_like(zones, dtype=bool)
    best: list[tuple[int, int]] = []
    for r in range(rows):
        for c in range(cols):
            if zones[r, c] != 0 or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < rows and 0 <= nc < cols and not seen[nr, nc] and zones[nr, nc] == 0:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            if len(comp) > len(best):
                best = comp
    return best


@dataclass
class Prediction:
    payload: dict[str, Any]


def predict_image(
    bgr: np.ndarray,
    gsd: float,
    model_bundle: dict[str, Any],
    th: dict[str, Any],
    max_cells: int = 2500,
) -> Prediction:
    """Run the trained patch classifier over a 50 m grid covering the image."""
    if gsd <= 0:
        raise ValueError("GSD must be positive")
    # Rescale so 50 m is always 100 px, matching the training resolution.
    # A 50 m cell covers 50/gsd pixels, so the resize factor is
    # 100 / (50/gsd) = 2*gsd = gsd / TRAIN_GSD.
    scale = float(gsd) / TRAIN_GSD
    if abs(scale - 1.0) > 1e-6:
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
        bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=interp)
    h, w = bgr.shape[:2]
    cols = max(1, int(round(w / CELL_PX)))
    rows = max(1, int(round(h / CELL_PX)))
    if rows * cols > max_cells:
        raise ValueError(f"Image covers {rows * cols} cells; limit is {max_cells}. Crop it or raise the GSD.")

    model = model_bundle["model"]
    probs = np.zeros((rows, cols, 3), dtype=np.float32)
    zones = np.full((rows, cols), 2, dtype=np.uint8)

    for r in range(rows):
        for c in range(cols):
            cx = (c + 0.5) * (w / cols)
            cy = (r + 0.5) * (h / rows)
            patch = padded_crop(bgr, cx, cy)
            feat = extract_features(patch).reshape(1, -1)
            p = model.predict_proba(feat)[0]
            # classes_ may be a subset; map to the canonical order.
            full = np.zeros(3, dtype=np.float32)
            for idx, cls in enumerate(model.classes_):
                full[int(cls)] = p[idx]
            probs[r, c] = full
            zones[r, c] = int(np.argmax(full))

    stats = {
        "safe_cells": int((zones == 0).sum()),
        "caution_cells": int((zones == 1).sum()),
        "unsafe_cells": int((zones == 2).sum()),
        "total_cells": int(zones.size),
    }
    stats["safe_pct"] = round(100.0 * stats["safe_cells"] / stats["total_cells"], 1)
    stats["caution_pct"] = round(100.0 * stats["caution_cells"] / stats["total_cells"], 1)
    stats["unsafe_pct"] = round(100.0 * stats["unsafe_cells"] / stats["total_cells"], 1)

    comp = _largest_safe_component(zones)
    best_cell = None
    if comp:
        # Prefer the safe cell with the highest mean safe probability, near the
        # centroid of the largest safe component.
        cr = sum(r for r, _ in comp) / len(comp)
        cc = sum(c for _, c in comp) / len(comp)
        best_r, best_c = max(comp, key=lambda rc: (probs[rc[0], rc[1], 0] - 0.002 * abs(rc[0] - cr) - 0.002 * abs(rc[1] - cc)))
        best_cell = {
            "row": int(best_r),
            "col": int(best_c),
            "safe_probability": round(float(probs[best_r, best_c, 0]), 3),
            "component_cells": len(comp),
            "component_area_m2": int(len(comp) * CELL_M * CELL_M),
        }

    if comp:
        verdict = "SAFE"
        headline = "SAFE TO LAND"
        recommendation = (
            f"At least one clear 50 m landing cell exists. The largest connected safe area is "
            f"{len(comp)} cell(s), about {len(comp) * 2500:,} m². Target the highlighted cell, then verify with "
            "higher-resolution imagery before committing."
        )
        confidence = float(probs[best_cell["row"], best_cell["col"], 0]) if best_cell else 0.0
    elif stats["caution_cells"] > 0:
        verdict = "CAUTION"
        headline = "CAUTION — SCOUT FIRST"
        recommendation = (
            "No fully safe 50 m cell was found, but caution-class terrain is present. "
            "Send a scout or inspect the caution cells at higher resolution; do not land blindly."
        )
        caution_idx = np.argwhere(zones == 1)
        confidence = float(np.max(probs[zones == 1][:, 1])) if len(caution_idx) else 0.0
    else:
        verdict = "UNSAFE"
        headline = "UNSAFE TO LAND"
        recommendation = (
            "Every 50 m cell in this image is classified unsafe for the lander profile. "
            "Choose a different landing site or re-image the area at higher resolution."
        )
        confidence = float(np.mean(probs[:, :, 2]))

    detections = _classical_detections(bgr, TRAIN_GSD, th)
    # Scale detections back to the original image if the input was rescaled.
    if abs(scale - 1.0) > 1e-6:
        inv = 1.0 / scale
        for d in detections:
            d["x"] = int(round(d["x"] * inv))
            d["y"] = int(round(d["y"] * inv))
            d["w"] = max(1, int(round(d["w"] * inv)))
            d["h"] = max(1, int(round(d["h"] * inv)))
            d["diameter_m"] = round(float(d["diameter_m"]), 2)

    training = model_bundle.get("training", {})
    payload = {
        "verdict": verdict,
        "headline": headline,
        "recommendation": recommendation,
        "confidence": round(float(confidence), 3),
        "worst_zone": int(zones.max()),
        "gsd_m": float(gsd),
        "assumed_gsd": True,
        "cell_size_m": CELL_M,
        "grid": {"rows": int(rows), "cols": int(cols), "zones": zones.astype(int).tolist()},
        "stats": stats,
        "best_cell": best_cell,
        "detections": detections,
        "detector": "classical",
        "model": {
            "name": training.get("model_name", "HOG + HistGradientBoosting"),
            "accuracy": training.get("accuracy"),
            "macro_f1": training.get("macro_f1"),
            "trained_on": training.get("trained_on", "synthetic labelled tiles"),
            "feature_version": FEATURE_VERSION,
        },
        "disclaimer": (
            "Demo only, not flight-certified. Safety uses the lander thresholds from thresholds.yaml "
            "and a 50 m grid. Uploads are assumed to be 0.5 m/px unless a GSD is supplied."
        ),
    }
    return Prediction(payload)
