# Phase Status

**Project:** Lunar Boulder Hazard — LROC NAC → detections → metric hazard map → Leaflet / upload / 3D viewers  
**Status:** End-to-end demo working on real Apollo 17 NAC imagery, including a trained landing-safety classifier for image uploads and a Three.js 3D site model. YOLOv8n training is blocked by a host security policy; the map detector falls back to labelled classical CV and is reported honestly.  
**Last verified:** 2026-11-13

---

## Phase summary

| Phase | Script | Status | Output |
|---|---|---|---|
| 0. Scaffold | `config.yaml`, `thresholds.yaml`, `requirements.txt` | Done | Pinned Python 3.12 environment, no-build static viewer |
| 1. Synthetic training data | `scripts/synthetic_data.py` | Done | 500 tiles, 7,495 boulder boxes, YOLO + COCO labels |
| 2. Real NAC fetch + tiling | `scripts/01_fetch_and_tile.py` | Done | 36 × 512 px tiles, 2,752 px mosaic, `tiles_metadata.json` |
| 3. YOLOv8n training | `scripts/02_train.py` | **Blocked on this machine** | Training path implemented; checkpoint not produced |
| 4. Detection + sizing + hazard | `scripts/03_infer_and_hazard.py` | Done (classical fallback) | 777 boulders, both profile grids, viewer JSON/PNG/TIF |
| 5. Landing-safety classifier | `scripts/04_train_safety_model.py` | Done, trained + evaluated | `landing_safety.joblib`, `safety_report.json` |
| 6. Upload API + page | `scripts/serve_demo.py`, `viewer/upload.html` | Done, browser-verified | `/api/predict`, SAFE/CAUTION/UNSAFE verdict, 50 m grid |
| 7. Leaflet viewer | `viewer/index.html` | Done, browser-verified | Interactive map, live hazard recompute, popups, URL state |
| 8. 3D site model | `viewer/3d.html` | Done, browser-verified | Three.js scene, controls, hover tooltips, `.glb` export |

---

## Real data

- **Site:** Apollo 17 — Taurus-Littrow, 20.1709° N, 30.6683° E.
- **Source:** USGS/NASA Apollo 17 LROC NAC landing-site orthomosaic, 0.5 m/px, 662 MB GeoTIFF read remotely with GDAL `/vsicurl/` range reads.
- **Tile grid:** 6 × 6 = 36 tiles, 512 px each, 64 px overlap.
- **Mosaic:** 2,752 × 2,752 px, local lunar Azimuthal Equidistant CRS in metres.
- **Exact GSD used for sizing:** 0.5 m/px (`rasterio` transform.a).
- **Metadata:** `data/labels/tiles_metadata.json` records source type `real_nac`, bounds, tile centres, CRS and GSD.

---

## Detection and hazard results (classical fallback)

- **Detector:** OpenCV top-hat bright-body + cast-shadow test; NMS; labelled `classical` everywhere.
- **Boulders:** 777 unique detections.
- **Diameter:** 2.0–6.0 m (median 2.5 m), estimated as longest box side × 0.5 m/px.
- **Confidence:** 0.275–0.88.
- **Hazard grid:** 28 × 28 = 784 cells at 50 m × 50 m.
- **Lander profile:** 342 safe · 182 caution · 260 unsafe.
- **Rover profile:** 342 safe · 0 caution · 442 unsafe.
- **Cells with detections:** 442.
- **Client/server parity:** the viewer recomputes the same grid in the browser and matches the exported Python stats exactly at the exported confidence (0.25).

The rover profile has no caution cells because its 1.5 m unsafe threshold is below every detected boulder (minimum 2.0 m); size therefore dominates the zone.

---

## Landing-safety classifier

PyTorch cannot load here, so a compact, genuinely trained classifier handles uploads
instead of faking a YOLO checkpoint.

- **Patches:** 500 synthetic tiles × 25 cells = 12,500 patches (50 m each, 0.5 m/px), labelled from the exact YOLO annotations using lander thresholds.
- **Split:** grouped 400 train / 100 test tiles (10,000 / 2,499 patches) — no tile leakage.
- **Features:** HOG + intensity histograms + edge/brightness statistics + bright-blob descriptors; 2,099 features (version 3).
- **Model:** `HistGradientBoostingClassifier`, class-balanced, 81 boosting iterations, ~109 s fit.
- **Accuracy:** 0.804; macro F1 0.747.
- **Per-class precision/recall:** safe 0.90/0.83 · caution 0.55/0.54 · unsafe 0.76/0.90.
- **Artifact:** `models/weights/landing_safety.joblib` (2.39 MB) + `safety_report.json`.
- **Spot checks:** per-tile cell accuracy 0.80–1.00 on four sample tiles; single-cell crops of dense terrain return UNSAFE with 0.60–0.93 confidence.
- **Serving:** FastAPI endpoint `POST /api/predict` rescales to the training resolution, classifies a 50 m grid, picks the largest connected safe component, and returns verdict/grid/stats/best cell/detections/model metadata.

Uploaded images are assumed to be lander-scale and 0.5 m/px unless a GSD is supplied.

---

## Viewer verification

### 2D Leaflet viewer (`index.html`)

Verified with headless Chrome and real mouse clicks at 1440 × 900:

- Basemap, hazard overlay and boulder markers render at metric scale.
- `L.CRS.Simple` scale is `2^zoom`; zoom range is −8 … 3, so 0 = 1 px/m and the scale bar matches the map.
- Lander ↔ Rover toggle updates zones, cards, coverage bar and URL.
- Confidence slider 0.05–0.95 filters markers and recomputes cells live.
- Hazard opacity slider 0–100% works; hazard overlay on/off works from both toggles.
- Boulder popup shows diameter, confidence, box size, GSD, detector, tile and zone.
- Hazard-cell popup shows max diameter, boulder count, profile, confidence and cell coordinates.
- Clicking an empty hazard cell works; marker clicks do not get replaced by cell popups.
- URL state works: `?profile=rover&conf=0.5&opacity=0.7&hazard=0`.
- About modal opens/closes; Escape closes it.
- New nav links to `/upload.html` and `/3d.html` work; navigating back restores the viewer and URL state.
- No console errors and no page errors in the final run.

### Upload page (`upload.html` + `/api/predict`)

Verified end-to-end with a real synthetic tile through the running server:

- Health check reports the trained model with accuracy/F1 and no notice.
- Drag/drop target accepts a PNG; clicking **Analyze** calls the API and returns a verdict card, 5×5 50 m grid, stats, target cell and detection list.
- A mostly-safe tile returns **SAFE TO LAND** with a highlighted 50 m target and 4-cell safe component.
- A dense single-cell crop returns **UNSAFE** (red verdict card).
- Detections shown on the image now use the same top-hat + cast-shadow method as the main pipeline, so sizes are plausible (no 30+ m terrain false positives).
- Clear resets the panel; no page/console errors.

### 3D site model (`3d.html`)

Verified with headless Chrome (WebGL via SwiftShader) and a headed-Chrome GLB export:

- Loads the real NAC texture, 784 hazard columns, 777 boulders and the lander target.
- Zone counts match the 2D export: lander 342/182/260; rover 342/0/442.
- Confidence 0.25 → 777 active boulders; 0.70 → 676, with matching stats.
- Vertical-exaggeration slider scales column heights (capped at 40 m); toggles hide/show columns and boulders.
- Top view, reset view and “Landing target” focus all move the camera; the focus camera chooses the clearest diagonal so hazard walls do not hide the lander.
- Hovering a boulder shows diameter, confidence and tile in a tooltip.
- **Download .glb** exports `apollo17_boulder_hazard.glb` (~5.7 MB, 5,692,324 bytes), verified completed in headed Chrome.
- No page/console errors in any run.

---

## Blocker: PyTorch / Smart App Control

This machine has Windows Smart App Control **On**. It blocks unsigned PyTorch DLLs:

```
OSError: [WinError 4551] An Application Control policy has blocked this file.
Error loading "...\torch\lib\caffe2_nvrtc.dll" (and shm.dll)
```

`torch==2.2.2+cu121` is installed but cannot be imported, so `scripts/02_train.py`
cannot train here. The pipeline does **not** fake a YOLO result:

- `03_infer_and_hazard.py` catches the load failure and uses the classical detector.
- `hazard_meta.json` records `"detector": "classical"` and the viewer shows **Classical CV**.
- `02_train.py` remains the real YOLOv8n path and fails with a clear message if PyTorch cannot load.
- The upload model is separately trained (HOG + HistGradientBoosting) and nothing labels it YOLOv8n.

Ways to get the trained result:

1. **Recommended:** run `scripts/02_train.py` on an unblocked machine or cloud GPU, then copy `models/weights/best.pt` back. `03_infer_and_hazard.py` prefers it automatically.
2. Disable Smart App Control manually (one-way security change; not done by this project).
3. Demo with the fallback detector as shipped; every downstream stage is still exercised.

ONNX Runtime 1.31.1 works on this machine, but no lunar boulder YOLO ONNX checkpoint was found; the available lunar models target craters or are PyTorch-only.

---

## Demo commands

```powershell
# from the project root
.\.venv\Scripts\python.exe scripts\synthetic_data.py
.\.venv\Scripts\python.exe scripts\01_fetch_and_tile.py
.\.venv\Scripts\python.exe scripts\04_train_safety_model.py
# optional / blocked here:
# .\.venv\Scripts\python.exe scripts\02_train.py
.\.venv\Scripts\python.exe scripts\03_infer_and_hazard.py --conf 0.25

# serves the 2D map, upload page/API and 3D model
.\.venv\Scripts\python.exe scripts\serve_demo.py
# http://localhost:8000/           2D hazard map
# http://localhost:8000/upload.html
# http://localhost:8000/3d.html
```

All pages must be served over HTTP, not opened as `file://`.

---

## Known limitations

- Classical map detection is a 2-D bright-object proxy, not a trained lunar boulder model; it under-detects dark rocks and can merge overlapping boulders.
- Diameter is the longest bounding-box side × GSD, not a 3-D shape reconstruction.
- Shadows, tile-edge clipping and overlapping rocks bias size estimates.
- The hazard grid uses maximum boulder size and detection count per cell; it does not model slope, terrain roughness or traverse cost.
- The upload classifier was trained on synthetic 512 px lunar-like tiles and only knows the lander profile; domain transfer to real NAC imagery is untested.
- The upload verdict assumes a 50 m grid and 0.5 m/px unless a GSD is supplied.
- The 3D scene is illustrative with a flat ground plane; there is no DEM, so it is not a photogrammetric reconstruction.
- This is a demo, not flight-certified decision support.
- `rio-cogeo` and `rio-tiler` are omitted from `requirements.txt`: the requested `rio-cogeo==4.1.0` does not exist and `rio-cogeo 4.0.1` conflicts with `rio-tiler 4.1.0`; neither is used by this pipeline.

---

## Next steps

- Train YOLOv8n on an unblocked GPU and re-run stage 04; the viewer will switch its badge to **YOLOv8n** automatically.
- Add manual boulder labels on a few real NAC tiles to measure domain-transfer mAP for both the map detector and the upload classifier.
- Test the upload model on cropped real NAC tiles and tune the grid/GSD assumptions from measured performance.
- Add a real DEM if stereo products become available, so the 3D terrain is no longer flat.
- Optionally package `viewer/` as a static site; the 2D map and 3D page have no server-side dependencies beyond the upload API.
