#!/usr/bin/env python
"""Stage 3 — train YOLOv8n on the synthetic lunar boulder dataset.

Builds a clean Ultralytics dataset from ``data/synthetic`` (85/15 split),
optionally adds unlabelled real tiles as an unseen domain test set, trains,
validates and copies the best checkpoint to ``models/weights/best.pt``.

Usage:
    python scripts/02_train.py
    python scripts/02_train.py --epochs 8 --batch 8   # quick smoke run
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from ruamel.yaml import YAML
yaml = YAML(typ="safe")

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def load_config() -> dict:
    with open(ROOT / "config.yaml", "r", encoding="utf-8") as fh:
        return yaml.load(fh)


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        dst.hardlink_to(src)
    except Exception:  # noqa: BLE001
        shutil.copy2(src, dst)


def build_dataset(cfg: dict, fraction: float) -> dict:
    syn_img = ROOT / "data" / "synthetic" / "images"
    syn_lbl = ROOT / "data" / "synthetic" / "labels"
    images = sorted(syn_img.glob("*.png"))
    if not images:
        raise SystemExit("No synthetic images found. Run: python scripts/synthetic_data.py")

    seed = int(cfg["project"].get("seed", 42))
    rng = random.Random(seed)
    rng.shuffle(images)
    if fraction < 1.0:
        images = images[: max(2, int(len(images) * fraction))]

    val_n = max(1, int(round(len(images) * float(cfg["synthetic"].get("val_split", 0.15)))))
    val = images[:val_n]
    train = images[val_n:]
    if not train:
        train, val = images[:1], images[1:] or images[:1]

    ds = ROOT / "data" / "train"
    if ds.exists():
        shutil.rmtree(ds)
    for split, files in (("train", train), ("val", val)):
        for img in files:
            lbl = syn_lbl / f"{img.stem}.txt"
            if not lbl.exists():
                continue
            link_or_copy(img, ds / "images" / split / img.name)
            link_or_copy(lbl, ds / "labels" / split / lbl.name)

    dataset = {
        "path": str(ds),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "boulder"},
    }
    with open(ds / "dataset.yaml", "w", encoding="utf-8") as fh:
        yaml.dump(dataset, fh, sort_keys=False)

    # Keep real tiles available as a domain-shift test set (never as negatives:
    # they may contain boulders and would teach the model to suppress them).
    real_dir = ROOT / "data" / "tiles" / "images"
    real_pngs = sorted(real_dir.glob("*.png"))
    test_ds = ROOT / "data" / "real_test"
    if test_ds.exists():
        shutil.rmtree(test_ds)
    test_ds.mkdir(parents=True, exist_ok=True)
    for p in real_pngs:
        link_or_copy(p, test_ds / p.name)

    return {
        "train": len(train),
        "val": len(val),
        "real_test": len(real_pngs),
        "dataset_yaml": str(ds / "dataset.yaml"),
        "dataset_dir": str(ds),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--name", default="yolov8n_lunar")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    mcfg = cfg["model"]
    epochs = args.epochs if args.epochs is not None else int(mcfg["epochs"])
    batch = args.batch if args.batch is not None else int(mcfg["batch_size"])
    imgsz = args.imgsz if args.imgsz is not None else int(mcfg["imgsz"])

    try:
        import torch
        from ultralytics import YOLO
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            "PyTorch/Ultralytics could not be loaded:\n"
            f"  {exc}\n"
            "Install the pinned environment first:\n"
            "  .venv\\Scripts\\pip install -r requirements.txt\n"
        )

    cuda = torch.cuda.is_available()
    device = args.device if args.device is not None else str(mcfg.get("device", 0))
    if not cuda:
        device = "cpu"
        if batch > 8:
            batch = 8
    if cuda:
        name = torch.cuda.get_device_name(0)
        free, total = torch.cuda.mem_get_info(0)
        print(f"[train] GPU: {name} ({free / 1e9:.1f} GB free / {total / 1e9:.1f} GB total)")
    else:
        print("[train] CUDA unavailable — training on CPU (reduce --epochs for a smoke test)")

    ds_info = build_dataset(cfg, args.fraction)
    print(f"[train] dataset: {ds_info['train']} train / {ds_info['val']} val, "
          f"{ds_info['real_test']} real test tiles")
    print(f"[train] {epochs} epochs, batch {batch}, imgsz {imgsz}, device {device}")

    run_name = args.name
    weights_dir = ROOT / "models" / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    run_dir = ROOT / "models" / "runs" / run_name
    t0 = time.time()

    model = None
    download_error = None
    try:
        model = YOLO("yolov8n.pt" if not args.resume else str(weights_dir / "best.pt"))
    except Exception as exc:  # noqa: BLE001
        download_error = str(exc)
        print(f"[train] pretrained weights unavailable ({exc}); training from scratch")
        model = YOLO("yolov8n.yaml")

    train_kwargs = dict(
        data=ds_info["dataset_yaml"],
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        lr0=float(mcfg["lr0"]),
        device=device,
        workers=int(mcfg.get("workers", 2)),
        patience=int(mcfg.get("patience", 12)),
        project=str(ROOT / "models" / "runs"),
        name=run_name,
        exist_ok=True,
        seed=int(cfg["project"].get("seed", 42)),
        deterministic=False,
        amp=bool(cuda),
        plots=True,
        verbose=True,
    )
    if args.resume:
        train_kwargs.pop("data", None)
        train_kwargs.pop("epochs", None)
        train_kwargs.pop("batch", None)
        train_kwargs.pop("imgsz", None)
        train_kwargs.pop("lr0", None)
        train_kwargs.pop("name", None)
        train_kwargs.pop("project", None)
        train_kwargs.pop("exist_ok", None)
        model.train(resume=True, **train_kwargs)
    else:
        try:
            model.train(**train_kwargs)
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower() or batch <= 2:
                raise
            new_batch = max(2, batch // 2)
            print(f"[train] CUDA OOM — retrying with batch {new_batch}")
            train_kwargs["batch"] = new_batch
            model = YOLO("yolov8n.pt")
            model.train(**train_kwargs)

    # Best checkpoint from this run.
    best = run_dir / "weights" / "best.pt"
    if not best.exists():
        # Ultralytics may increment the run directory suffix.
        candidates = sorted(ROOT.glob(f"models/runs/{run_name}*/weights/best.pt"),
                            key=lambda p: p.stat().st_mtime)
        if candidates:
            best = candidates[-1]
            run_dir = best.parents[1]
    if not best.exists():
        raise SystemExit("Training finished but best.pt was not found")
    shutil.copy2(best, weights_dir / "best.pt")
    print(f"[train] best weights → {weights_dir / 'best.pt'}")

    metrics: dict = {}
    try:
        val_model = YOLO(str(weights_dir / "best.pt"))
        results = val_model.val(data=ds_info["dataset_yaml"], device=device,
                                conf=0.001, iou=0.6, plots=False)
        metrics = {k: float(v) for k, v in results.results_dict.items()}
    except Exception as exc:  # noqa: BLE001
        print(f"[train] validation metrics unavailable: {exc}")

    report = {
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
        "epochs": epochs,
        "batch": batch,
        "imgsz": imgsz,
        "elapsed_s": round(time.time() - t0, 1),
        "dataset": ds_info,
        "weights": str(weights_dir / "best.pt"),
        "run_dir": str(run_dir),
        "pretrained_weights_error": download_error,
        "metrics": metrics,
    }
    with open(weights_dir / "training_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    print(f"\n✓ Training complete in {report['elapsed_s'] / 60:.1f} min")
    if metrics:
        print(f"  mAP50:    {metrics.get('metrics/mAP50(B)', 0):.4f}")
        print(f"  mAP50-95: {metrics.get('metrics/mAP50-95(B)', 0):.4f}")
        print(f"  precision:{metrics.get('metrics/precision(B)', 0):.4f}")
        print(f"  recall:   {metrics.get('metrics/recall(B)', 0):.4f}")
    print(f"  report:   {weights_dir / 'training_report.json'}")


if __name__ == "__main__":
    main()
