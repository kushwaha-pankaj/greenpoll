#!/usr/bin/env python3
"""Step 4 — CrossPoll joint pretraining on apple+strawberry+tomato.

Trains YOLOv8n on the harmonized multi-crop flower dataset (single
'flower' class) starting from COCO pretrained weights. The resulting
checkpoint is the CrossPoll source-domain feature extractor that
run_baselines.py --methods crosspoll fine-tunes on kiwi.

Prereqs:
  - Run scripts/harmonize_labels.py first to build data/joint/.
  - Set GREENPOLL_DEVICE=0 on Kaggle/CUDA, or rely on auto-detect.

Usage:
    python scripts/pretrain_joint.py
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
JOINT_DIR = ROOT / "data" / "joint"
RUN_DIR = ROOT / "runs" / "joint"
CONFIG_PATH = ROOT / "configs" / "experiments" / "crosspoll.yaml"

# Source crops to pretrain on (kiwi is held out for fine-tuning)
SOURCE_CROPS = ["apple", "strawberry", "tomato"]


def resolve_device(cfg_device: str | None) -> int | str | None:
    env = os.environ.get("GREENPOLL_DEVICE")
    if env:
        return int(env) if env.isdigit() else env
    if cfg_device == "mps" and torch.backends.mps.is_available():
        return "mps"
    if cfg_device == "mps" and torch.cuda.is_available():
        return 0
    if cfg_device and cfg_device != "mps":
        return cfg_device
    return None


def build_data_yaml() -> Path:
    """Write the joint dataset yaml pointing at the harmonized splits."""
    train = [str(JOINT_DIR / c / "train" / "images") for c in SOURCE_CROPS
             if (JOINT_DIR / c / "train" / "images").exists()]
    val = [str(JOINT_DIR / c / "valid" / "images") for c in SOURCE_CROPS
           if (JOINT_DIR / c / "valid" / "images").exists()]
    if not train:
        raise FileNotFoundError(
            f"No joint train splits found under {JOINT_DIR}. "
            "Run scripts/harmonize_labels.py first."
        )
    data = {"train": train, "val": val, "nc": 1, "names": ["flower"]}
    out = RUN_DIR / "data.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        yaml.dump(data, f, default_flow_style=False)
    return out


def main() -> None:
    cfg = yaml.safe_load(open(CONFIG_PATH))
    t = cfg["training"]
    device = resolve_device(t.get("device"))

    data_yaml = build_data_yaml()
    print(f"Joint data yaml: {data_yaml}")

    model = YOLO("yolov8n.pt")  # start from COCO; richer init than scratch
    t0 = time.time()
    train_kwargs = dict(
        data=str(data_yaml),
        epochs=t["epochs"],
        patience=t["patience"],
        imgsz=t["imgsz"],
        batch=t["batch"],
        optimizer=t["optimizer"],
        amp=t["amp"],
        lr0=t["lr0_joint"],         # higher LR — pretraining, not fine-tuning
        cache=t.get("cache", False),
        cos_lr=t.get("cos_lr", False),
        workers=t.get("workers", 4),
        deterministic=t.get("deterministic", False),
        plots=t.get("plots", False),
        seed=cfg["reproducibility"]["seeds"][0],
        project=str(RUN_DIR),
        name="pretrain",
        exist_ok=True,
        verbose=True,
    )
    if device is not None:
        train_kwargs["device"] = device

    model.train(**train_kwargs)
    elapsed_min = (time.time() - t0) / 60.0

    best = RUN_DIR / "pretrain" / "weights" / "best.pt"
    print(f"\n✓ Joint pretrain done in {elapsed_min:.1f} min")
    print(f"  Best ckpt: {best}")
    print(f"  Next:      python scripts/run_baselines.py --methods crosspoll")


if __name__ == "__main__":
    main()
