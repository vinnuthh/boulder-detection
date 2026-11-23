"""Train the landing-safety classifier on the labelled synthetic dataset.

PyTorch/YOLOv8n cannot load on this Windows machine because Smart App Control
blocks unsigned CUDA DLLs.  Instead of faking a checkpoint, this script trains
a compact HOG + HistGradientBoosting model on 50 m patches cut from the 500
synthetic tiles.  Labels come from the exact YOLO annotations and the lander
thresholds in thresholds.yaml.

Run:
    python scripts/04_train_safety_model.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from safety_model import (  # noqa: E402
    CLASS_NAMES,
    FEATURE_VERSION,
    MODEL_PATH,
    REPORT_PATH,
    ROOT,
    cell_labels_from_yolo,
    extract_features,
    load_thresholds,
    padded_crop,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

IMAGES = ROOT / "data" / "synthetic" / "images"
LABELS = ROOT / "data" / "synthetic" / "labels"
IMAGE_PX = 512
GSD = 0.5
CELLS = 5
CELL_PX = IMAGE_PX / CELLS  # 102.4 px = 51.2 m


def build_dataset(th: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    images = sorted(IMAGES.glob("*.png"))
    if not images:
        raise SystemExit(f"No synthetic tiles in {IMAGES}; run scripts/synthetic_data.py first.")
    X: list[np.ndarray] = []
    y: list[int] = []
    groups: list[str] = []
    for img_path in tqdm(images, desc="patches", unit="tile"):
        label_path = LABELS / (img_path.stem + ".txt")
        if not label_path.exists():
            continue
        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        zones = cell_labels_from_yolo(label_path, IMAGE_PX, GSD, th, CELLS)
        for r in range(CELLS):
            for c in range(CELLS):
                cx = (c + 0.5) * CELL_PX
                cy = (r + 0.5) * CELL_PX
                patch = padded_crop(bgr, cx, cy)
                X.append(extract_features(patch))
                y.append(zones[r * CELLS + c])
                groups.append(img_path.stem)
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int32), np.asarray(groups)


def main() -> None:
    started = time.time()
    thresholds = load_thresholds()
    th = thresholds["profiles"]["lander"]

    print(f"[train] synthetic tiles: {IMAGES}")
    print(f"[train] labels: lander profile, {CELLS}x{CELLS} cells, {CELL_PX:.1f} px per cell")
    X, y, groups = build_dataset(th)
    print(f"[train] feature matrix: {X.shape[0]:,} patches x {X.shape[1]:,} features")

    split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(split.split(X, y, groups))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    print(f"[train] groups: {len(set(groups[train_idx]))} train / {len(set(groups[test_idx]))} test tiles")
    print(f"[train] patches: {len(train_idx):,} train / {len(test_idx):,} test")

    clf = HistGradientBoostingClassifier(
        learning_rate=0.07,
        max_iter=400,
        max_leaf_nodes=31,
        min_samples_leaf=15,
        l2_regularization=1.0,
        class_weight="balanced",
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=30,
        random_state=42,
        verbose=1,
    )
    t0 = time.time()
    clf.fit(X_train, y_train)
    train_s = time.time() - t0
    print(f"[train] fit in {train_s:.1f}s ({clf.n_iter_} boosting iterations)")

    pred = clf.predict(X_test)
    acc = float(accuracy_score(y_test, pred))
    macro_f1 = float(f1_score(y_test, pred, average="macro"))
    report = classification_report(
        y_test, pred, labels=[0, 1, 2], target_names=list(CLASS_NAMES), output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_test, pred, labels=[0, 1, 2]).astype(int).tolist()
    print(f"[train] accuracy {acc:.3f} | macro F1 {macro_f1:.3f}")
    print(classification_report(y_test, pred, labels=[0, 1, 2], target_names=list(CLASS_NAMES), zero_division=0))

    training = {
        "model_name": "HOG + HistGradientBoosting",
        "trained_on": "500 synthetic 512 px tiles (0.5 m/px), 50 m patches, exact YOLO labels",
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "n_train_patches": int(len(train_idx)),
        "n_test_patches": int(len(test_idx)),
        "n_train_tiles": int(len(set(groups[train_idx]))),
        "n_test_tiles": int(len(set(groups[test_idx]))),
        "n_features": int(X.shape[1]),
        "n_boosting_iterations": int(clf.n_iter_),
        "fit_seconds": round(train_s, 1),
        "confusion_matrix": cm,
        "label_order": list(CLASS_NAMES),
        "per_class": {
            name: {
                "precision": round(float(report[name]["precision"]), 4),
                "recall": round(float(report[name]["recall"]), 4),
                "f1": round(float(report[name]["f1-score"]), 4),
                "support": int(report[name]["support"]),
            }
            for name in CLASS_NAMES
        },
        "thresholds": th,
        "feature_version": FEATURE_VERSION,
        "note": (
            "PyTorch is blocked by Windows Smart App Control, so this is a trained "
            "classical-vision model, not YOLOv8n. The 2D/3D map still shows the "
            "classical boulder detector and its metric hazard grid."
        ),
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": clf,
            "feature_version": FEATURE_VERSION,
            "classes": list(CLASS_NAMES),
            "training": training,
            "thresholds": th,
        },
        MODEL_PATH,
        compress=3,
    )
    REPORT_PATH.write_text(json.dumps(training, indent=2), encoding="utf-8")
    size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
    print(f"[train] saved {MODEL_PATH} ({size_mb:.2f} MB)")
    print(f"[train] saved {REPORT_PATH}")
    print(f"[done] {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
