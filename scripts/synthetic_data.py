#!/usr/bin/env python
"""Stage 1 — generate a synthetic LROC-like boulder dataset.

Outputs
-------
data/synthetic/images/img_XXXX.png   512x512 grayscale tiles
data/synthetic/labels/img_XXXX.txt   YOLO boxes (class 0 = boulder)
data/labels/instances.json           COCO annotations + per-boulder metadata
data/labels/synthetic_stats.json     dataset summary used by later stages
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from ruamel.yaml import YAML
yaml = YAML(typ="safe")
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lunar_synth import render_scene, to_uint8  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def load_config() -> dict:
    with open(ROOT / "config.yaml", "r", encoding="utf-8") as fh:
        return yaml.load(fh)


def main() -> None:
    cfg = load_config()
    syn = cfg["synthetic"]
    gsd = float(cfg["tiling"]["target_gsd"])
    size = int(syn.get("image_size", 512))
    num_images = int(syn["num_images"])
    seed = int(cfg["project"].get("seed", 42))

    img_dir = ROOT / "data" / "synthetic" / "images"
    lbl_dir = ROOT / "data" / "synthetic" / "labels"
    meta_dir = ROOT / "data" / "labels"
    for d in (img_dir, lbl_dir, meta_dir):
        d.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    coco = {
        "info": {
            "description": "Synthetic lunar boulder dataset for YOLOv8n",
            "version": "1.0",
            "gsd_m_per_px": gsd,
            "site": cfg["project"]["site"],
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "boulder", "supercategory": "rock"},
        ],
    }
    ann_id = 1
    diameters_m: list[float] = []
    total_boxes = 0
    t0 = time.time()

    for i in tqdm(range(num_images), desc="synthetic tiles", unit="img"):
        img_rng = np.random.default_rng(seed + i * 7919)
        n_b = int(img_rng.integers(
            int(syn["boulders_per_image"]["min"]),
            int(syn["boulders_per_image"]["max"]) + 1,
        ))
        scene = render_scene(
            size,
            img_rng,
            n_boulders=n_b,
            boulder_size_px=(
                float(syn["boulder_size_px"]["min"]),
                float(syn["boulder_size_px"]["max"]),
            ),
            n_craters=int(img_rng.integers(
                int(syn["craters_per_image"]["min"]),
                int(syn["craters_per_image"]["max"]) + 1,
            )),
            n_micro_rocks=int(img_rng.integers(
                int(syn["background_rocks"]["min"]),
                int(syn["background_rocks"]["max"]) + 1,
            )),
        )

        name = f"img_{i:04d}"
        png_path = img_dir / f"{name}.png"
        cv2.imwrite(str(png_path), to_uint8(scene.image))

        image_id = i + 1
        coco["images"].append({
            "id": image_id,
            "file_name": f"{name}.png",
            "width": size,
            "height": size,
            "gsd_m_per_px": gsd,
            "source": "synthetic",
        })

        label_lines: list[str] = []
        for b in scene.boulders:
            bw, bh = b.x2 - b.x1 + 1, b.y2 - b.y1 + 1
            if bw < 2 or bh < 2:
                continue
            cx_n = (b.x1 + b.x2) / 2.0 / size
            cy_n = (b.y1 + b.y2) / 2.0 / size
            w_n = bw / size
            h_n = bh / size
            label_lines.append(f"0 {cx_n:.6f} {cy_n:.6f} {w_n:.6f} {h_n:.6f}")
            diameter_m = max(bw, bh) * gsd
            diameters_m.append(diameter_m)
            total_boxes += 1
            coco["annotations"].append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": 1,
                "bbox": [b.x1, b.y1, bw, bh],
                "area": bw * bh,
                "iscrowd": 0,
                "diameter_px": float(max(bw, bh)),
                "diameter_m": round(diameter_m, 3),
                "center_px": [round(b.cx, 2), round(b.cy, 2)],
            })
            ann_id += 1

        (lbl_dir / f"{name}.txt").write_text("\n".join(label_lines) + ("\n" if label_lines else ""),
                                             encoding="utf-8")

    with open(meta_dir / "instances.json", "w", encoding="utf-8") as fh:
        json.dump(coco, fh, indent=1)

    stats = {
        "images": num_images,
        "annotations": total_boxes,
        "synthetic_image_size_px": size,
        "gsd_m_per_px": gsd,
        "diameter_m": {
            "min": round(float(np.min(diameters_m)), 3) if diameters_m else 0,
            "max": round(float(np.max(diameters_m)), 3) if diameters_m else 0,
            "mean": round(float(np.mean(diameters_m)), 3) if diameters_m else 0,
            "p50": round(float(np.median(diameters_m)), 3) if diameters_m else 0,
            "p90": round(float(np.percentile(diameters_m, 90)), 3) if diameters_m else 0,
        },
        "elapsed_s": round(time.time() - t0, 1),
        "seed": seed,
    }
    with open(meta_dir / "synthetic_stats.json", "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)

    # Contact sheet makes it obvious in one glance whether the data is sane.
    preview_n = min(9, num_images)
    thumbs = []
    for i in range(preview_n):
        im = cv2.imread(str(img_dir / f"img_{i:04d}.png"), cv2.IMREAD_GRAYSCALE)
        thumbs.append(cv2.resize(im, (256, 256), interpolation=cv2.INTER_AREA))
    if thumbs:
        while len(thumbs) < 9:
            thumbs.append(np.zeros((256, 256), np.uint8))
        rows = [np.hstack(thumbs[r * 3:(r + 1) * 3]) for r in range(3)]
        cv2.imwrite(str(meta_dir / "synthetic_preview.jpg"), np.vstack(rows))

    print(f"\n✓ Synthetic dataset: {num_images} tiles, {total_boxes} boulders")
    print(f"  diameters: {stats['diameter_m']['min']}–{stats['diameter_m']['max']} m "
          f"(p50 {stats['diameter_m']['p50']} m)")
    print(f"  images: {img_dir}")
    print(f"  labels: {meta_dir / 'instances.json'}")
    print(f"  elapsed: {stats['elapsed_s']} s")


if __name__ == "__main__":
    main()
