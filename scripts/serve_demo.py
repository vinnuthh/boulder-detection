"""Local demo server: static viewer + landing-safety upload API.

Run:
    python scripts/serve_demo.py
    # open http://localhost:8000/

The API is intentionally small and local-only.  When the scikit-learn model
cannot be loaded (e.g. Windows Smart App Control blocks scipy DLLs), the
server starts in classical-only mode: it still classifies 50 m cells from the
bright-body / shadow detector alone so uploads remain functional.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent))
from safety_model import (  # noqa: E402
    FEATURE_VERSION,
    MODEL_PATH,
    ROOT,
    decode_data_url,
    load_thresholds,
    predict_image,
    _classical_detections,
    zone_from_values,
    CELL_M,
    CELL_PX,
    TRAIN_GSD,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

VIEWER_DIR = ROOT / "viewer"
MAX_SIDE = 2400
_LOADED: dict[str, Any] | None = None
_LOAD_ERROR: str | None = None
CLASSICAL_ONLY = False


def get_model() -> dict[str, Any] | None:
    global _LOADED, _LOAD_ERROR, CLASSICAL_ONLY
    if _LOADED is not None:
        return _LOADED
    if _LOAD_ERROR is not None and not CLASSICAL_ONLY:
        raise HTTPException(503, _LOAD_ERROR)
    if not MODEL_PATH.exists():
        _LOAD_ERROR = (
            f"Model not found at {MODEL_PATH}. Run scripts/04_train_safety_model.py first."
        )
        CLASSICAL_ONLY = True
        return None
    try:
        import joblib  # local import so scipy failure doesn't kill startup
        bundle = joblib.load(MODEL_PATH)
    except Exception as exc:  # pragma: no cover - defensive
        _LOAD_ERROR = f"Could not load the safety model: {exc}"
        CLASSICAL_ONLY = True
        print(f"[serve] falling back to classical-only mode: {exc}")
        return None
    if int(bundle.get("feature_version", -1)) != FEATURE_VERSION:
        _LOAD_ERROR = (
            "The saved model uses an older feature version. Re-run "
            "scripts/04_train_safety_model.py."
        )
        CLASSICAL_ONLY = True
        return None
    _LOADED = bundle
    return bundle


class PredictRequest(BaseModel):
    image: str = Field(..., description="data:image/...;base64,... payload")
    gsd: float = Field(0.5, gt=0.01, le=20.0, description="metres per pixel")


app = FastAPI(title="Lunar Landing Safety", version="1.0", docs_url="/api/docs")


def _classical_predict(bgr: np.ndarray, gsd: float, th: dict[str, Any]) -> dict[str, Any]:
    """Pure classical fallback: cell zones from bright-body detector alone."""
    # A 50 m cell covers CELL_M / gsd pixels, not a fixed CELL_PX, so that the
    # grid stays metric when the user supplies a GSD other than 0.5 m/px.
    cell_px = max(1.0, CELL_M / float(gsd))
    cols = max(1, int(round(bgr.shape[1] / cell_px)))
    rows = max(1, int(round(bgr.shape[0] / cell_px)))
    if rows * cols > 2500:
        raise ValueError(f"Image covers {rows * cols} cells; limit is 2500. Crop it or raise the GSD.")

    detections = _classical_detections(bgr, gsd, th)
    zones = np.zeros((rows, cols), dtype=np.uint8)

    # Bin detections into cells
    for d in detections:
        cx = d["x"] + d["w"] / 2
        cy = d["y"] + d["h"] / 2
        c = int(min(cols - 1, cx / cell_px))
        r = int(min(rows - 1, cy / cell_px))
        key = (r, c)
        if zones[key] < 2:
            # escalate based on this single boulder's size
            zones[key] = zone_from_values(d["diameter_m"], 1, th)

    # Also add density penalty: if >3 boulders in a cell, bump to at least caution
    cell_counts: dict[tuple[int, int], int] = {}
    for d in detections:
        cx = d["x"] + d["w"] / 2
        cy = d["y"] + d["h"] / 2
        c = int(min(cols - 1, cx / cell_px))
        r = int(min(rows - 1, cy / cell_px))
        cell_counts[(r, c)] = cell_counts.get((r, c), 0) + 1
    for (r, c), cnt in cell_counts.items():
        if cnt >= th["density"]["unsafe_min_count"]:
            zones[r, c] = 2
        elif cnt > th["density"]["safe_max_count"] and zones[r, c] < 1:
            zones[r, c] = 1

    stats = {
        "safe_cells": int((zones == 0).sum()),
        "caution_cells": int((zones == 1).sum()),
        "unsafe_cells": int((zones == 2).sum()),
        "total_cells": int(zones.size),
    }
    stats["safe_pct"] = round(100.0 * stats["safe_cells"] / stats["total_cells"], 1)
    stats["caution_pct"] = round(100.0 * stats["caution_cells"] / stats["total_cells"], 1)
    stats["unsafe_pct"] = round(100.0 * stats["unsafe_cells"] / stats["total_cells"], 1)

    # Best cell = first safe cell
    best_cell = None
    safe_cells = np.argwhere(zones == 0)
    if len(safe_cells):
        r, c = map(int, safe_cells[0])
        best_cell = {
            "row": r,
            "col": c,
            "safe_probability": 0.85,
            "component_cells": int(stats["safe_cells"]),
            "component_area_m2": int(stats["safe_cells"] * 2500),
        }

    if stats["safe_cells"] > 0:
        verdict = "SAFE"
        headline = "SAFE TO LAND"
        recommendation = (
            f"At least one clear 50 m landing cell exists (classical detector only). "
            f"Largest connected safe area: {stats['safe_cells']} cell(s), "
            f"about {stats['safe_cells'] * 2500:,} m². "
            "Note: running without trained ML model due to environment restrictions."
        )
        confidence = 0.85
    elif stats["caution_cells"] > 0:
        verdict = "CAUTION"
        headline = "CAUTION — SCOUT FIRST"
        recommendation = (
            "No fully safe 50 m cell found by classical detector. "
            "Send a scout or inspect at higher resolution before landing."
        )
        confidence = 0.6
    else:
        verdict = "UNSAFE"
        headline = "UNSAFE TO LAND"
        recommendation = (
            "Every 50 m cell is classified unsafe by classical detector. "
            "Choose a different landing site or re-image at higher resolution."
        )
        confidence = 0.9

    return {
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
        "detections": detections[:400],
        "detector": "classical",
        "model": {
            "name": "Classical CV fallback",
            "accuracy": None,
            "macro_f1": None,
            "trained_on": "none (fallback mode)",
            "feature_version": FEATURE_VERSION,
        },
        "disclaimer": (
            "Demo only, not flight-certified. Running in classical-only fallback mode "
            "because the trained ML model is unavailable in this environment. "
            "Hazard logic from thresholds.yaml."
        ),
    }


@app.get("/api/health")
@app.get("/health", include_in_schema=False)
def health() -> dict[str, Any]:
    bundle = get_model()
    training = bundle.get("training", {}) if bundle else {}
    return {
        "ok": not CLASSICAL_ONLY,
        "classical_only": CLASSICAL_ONLY,
        "model": training.get("model_name") or "Classical CV fallback",
        "accuracy": training.get("accuracy"),
        "macro_f1": training.get("macro_f1"),
        "feature_version": FEATURE_VERSION,
        "error": _LOAD_ERROR,
    }


@app.get("/api/model")
def model_info() -> dict[str, Any]:
    bundle = get_model()
    return {
        "training": bundle.get("training", {}) if bundle else {},
        "thresholds": load_thresholds()["profiles"]["lander"],
        "feature_version": FEATURE_VERSION,
        "classical_only": CLASSICAL_ONLY,
    }


@app.post("/api/predict")
def predict(req: PredictRequest) -> dict[str, Any]:
    if len(req.image) > 24_000_000:
        raise HTTPException(413, "Image payload is too large (limit ~18 MB binary).")
    try:
        bgr = decode_data_url(req.image)
    except Exception as exc:
        raise HTTPException(400, f"Invalid image: {exc}") from exc

    h, w = bgr.shape[:2]
    resized = False
    gsd = float(req.gsd)
    scale = 1.0
    longest = max(h, w)
    if longest > MAX_SIDE:
        scale = MAX_SIDE / float(longest)
        bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        gsd = gsd / scale
        resized = True

    th = load_thresholds()["profiles"]["lander"]
    bundle = get_model()
    if bundle is None:
        prediction = _classical_predict(bgr, gsd, th)
    else:
        try:
            prediction = predict_image(bgr, gsd, bundle, th).payload
        except Exception as exc:
            prediction = _classical_predict(bgr, gsd, th)
            prediction["disclaimer"] += f" (ML predict failed: {exc})"

    prediction["processed"] = {
        "width": int(bgr.shape[1]),
        "height": int(bgr.shape[0]),
        "resized": resized,
        "original_width": int(w),
        "original_height": int(h),
        "effective_gsd_m": round(float(gsd), 4),
    }
    prediction["model"]["thresholds"] = th
    return prediction


# Static viewer is mounted last so /api routes win.
app.mount("/", StaticFiles(directory=str(VIEWER_DIR), html=True), name="viewer")


if __name__ == "__main__":
    bundle = get_model()
    if bundle is not None:
        print("[serve] landing-safety model loaded")
    else:
        print("[serve] running in classical-only fallback mode")
    print("[serve] http://localhost:8000/  (2D map)")
    print("[serve] http://localhost:8000/upload.html  (image upload)")
    print("[serve] http://localhost:8000/3d.html  (3D site model)")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
