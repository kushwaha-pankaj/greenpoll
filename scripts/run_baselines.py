#!/usr/bin/env python3
"""Step 3 — Run Baseline #1 (Scratch) and Baseline #2 (COCO Transfer) on kiwi.

For each (method, seed, label_budget):
  1. Subsample kiwi train images to label_budget size
  2. Create a temporary data.yaml with subsampled paths
  3. Train YOLOv8n (from scratch or COCO pretrained)
  4. Evaluate on kiwi test set
  5. Append results to results/summary.csv and experiments/registry.csv

Usage:
    # Sanity check — 1 seed, 1 budget
    python scripts/run_baselines.py --seeds 42 --budgets 25

    # Full run — all seeds × all budgets
    python scripts/run_baselines.py
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
import time
from datetime import datetime
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

# ── Paths ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "kiwi"
RESULTS_CSV = ROOT / "results" / "summary.csv"
REGISTRY_CSV = ROOT / "experiments" / "registry.csv"
CONFIG_PATH = ROOT / "configs" / "experiments" / "crosspoll.yaml"
RUNS_DIR = ROOT / "runs"

# ── Load experiment config ─────────────────────────────────────────────
with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)

DEFAULT_SEEDS = CFG["reproducibility"]["seeds"]
DEFAULT_BUDGETS = CFG["data"]["label_budgets"]
EPOCHS = CFG["training"]["epochs"]
PATIENCE = CFG["training"]["patience"]
IMGSZ = CFG["training"]["imgsz"]
BATCH = CFG["training"]["batch"]
OPTIMIZER = CFG["training"]["optimizer"]
AMP = CFG["training"]["amp"]
LR_FINETUNE = CFG["training"]["lr0_finetune"]
CACHE = CFG["training"].get("cache", False)
COS_LR = CFG["training"].get("cos_lr", False)
WORKERS = CFG["training"].get("workers", 8)
DETERMINISTIC = CFG["training"].get("deterministic", True)
PLOTS = CFG["training"].get("plots", True)

# ── Device selection ───────────────────────────────────────────────────
# Priority: GREENPOLL_DEVICE env var > YAML config > auto-detect.
# Set GREENPOLL_DEVICE=0 on Kaggle/Colab CUDA boxes; "mps" on Apple Silicon;
# "cpu" to force CPU. An int-like string ("0", "0,1") is passed through to
# Ultralytics, which accepts CUDA indices.
_env_device = os.environ.get("GREENPOLL_DEVICE")
_cfg_device = CFG["training"].get("device", None)
if _env_device:
    DEVICE = int(_env_device) if _env_device.isdigit() else _env_device
elif _cfg_device == "mps" and torch.backends.mps.is_available():
    DEVICE = "mps"
elif _cfg_device == "mps" and torch.cuda.is_available():
    DEVICE = 0  # YAML asked for MPS but we're on a CUDA host (e.g. Kaggle)
elif _cfg_device and _cfg_device != "mps":
    DEVICE = _cfg_device
else:
    DEVICE = None  # let Ultralytics auto-detect


def list_image_label_pairs(split_dir: Path) -> list[tuple[Path, Path]]:
    """Return sorted (image, label) pairs for a split directory."""
    img_dir = split_dir / "images"
    lbl_dir = split_dir / "labels"
    pairs = []
    for img in sorted(img_dir.iterdir()):
        if img.suffix.lower() in (".jpg", ".jpeg", ".png"):
            lbl = lbl_dir / (img.stem + ".txt")
            if lbl.exists():
                pairs.append((img, lbl))
    return pairs


def load_completed_run_ids() -> set[str]:
    """Load run_ids already recorded in summary.csv to skip reruns."""
    completed = set()
    if RESULTS_CSV.exists():
        with open(RESULTS_CSV) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("run_id"):
                    completed.add(row["run_id"])
    return completed


def subsample_dataset(
    pairs: list[tuple[Path, Path]],
    budget: int,
    seed: int,
    dest: Path,
) -> Path:
    """Create a subsampled dataset at dest with symlinks. Returns dest path."""
    rng = random.Random(seed)
    n = min(budget, len(pairs))
    selected = rng.sample(pairs, n)

    img_out = dest / "images"
    lbl_out = dest / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    for img, lbl in selected:
        os.symlink(img, img_out / img.name)
        os.symlink(lbl, lbl_out / lbl.name)

    return dest


def build_data_yaml(
    train_dir: Path,
    val_dir: Path,
    test_dir: Path,
    nc: int,
    names: list[str],
    out_path: Path,
) -> Path:
    """Write a YOLO data.yaml with absolute paths."""
    data = {
        "train": str(train_dir / "images"),
        "val": str(val_dir / "images"),
        "test": str(test_dir / "images"),
        "nc": nc,
        "names": names,
    }
    with open(out_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)
    return out_path


def run_single_experiment(
    method: str,
    seed: int,
    budget: int,
    dry_run: bool = False,
) -> dict:
    """Train one model and return metrics dict."""
    run_id = f"{method}_kiwi_b{budget}_s{seed}"
    print(f"\n{'='*60}")
    print(f"  {run_id}")
    print(f"  method={method}  budget={budget}  seed={seed}")
    print(f"{'='*60}")

    actual_budget = min(budget, 181)  # kiwi train has 181 images
    if actual_budget < budget:
        print(f"  ⚠ Budget {budget} > available (181), capping to {actual_budget}")

    # ── Subsample training data ────────────────────────────────────
    all_pairs = list_image_label_pairs(DATA_DIR / "train")
    tmp_dir = RUNS_DIR / "tmp" / run_id
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    subsample_dataset(all_pairs, actual_budget, seed, tmp_dir / "train")

    # ── Build data.yaml ────────────────────────────────────────────
    data_yaml = tmp_dir / "data.yaml"
    build_data_yaml(
        train_dir=tmp_dir / "train",
        val_dir=DATA_DIR / "valid",
        test_dir=DATA_DIR / "test",
        nc=1,
        names=["kiwi-flower"],
        out_path=data_yaml,
    )

    if dry_run:
        print(f"  [DRY RUN] Would train {run_id}")
        shutil.rmtree(tmp_dir)
        return {}

    # ── Create model ───────────────────────────────────────────────
    if method == "scratch":
        model = YOLO("yolov8n.yaml")  # architecture only, random init
    elif method == "coco_transfer":
        model = YOLO("yolov8n.pt")  # COCO pretrained
    else:
        raise ValueError(f"Unknown method: {method}")

    # ── Train ──────────────────────────────────────────────────────
    project_dir = RUNS_DIR / "step3" / method
    t0 = time.time()
    train_kwargs = dict(
        data=str(data_yaml),
        epochs=EPOCHS,
        patience=PATIENCE,
        imgsz=IMGSZ,
        batch=BATCH,
        optimizer=OPTIMIZER,
        amp=AMP,
        lr0=LR_FINETUNE,
        cache=CACHE,
        cos_lr=COS_LR,
        workers=WORKERS,
        deterministic=DETERMINISTIC,
        plots=PLOTS,
        seed=seed,
        project=str(project_dir),
        name=run_id,
        exist_ok=True,
        verbose=True,
    )
    if DEVICE is not None:
        train_kwargs["device"] = DEVICE
    results = model.train(**train_kwargs)
    train_time = (time.time() - t0) / 60.0  # minutes

    # ── Evaluate on test set ───────────────────────────────────────
    best_ckpt = project_dir / run_id / "weights" / "best.pt"
    eval_model = YOLO(str(best_ckpt))
    metrics = eval_model.val(data=str(data_yaml), split="test")

    mAP50 = float(metrics.box.map50)
    precision = float(metrics.box.mp)
    recall = float(metrics.box.mr)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # ── Model size / latency ───────────────────────────────────────
    model_size_mb = best_ckpt.stat().st_size / (1024 * 1024)

    row = {
        "run_id": run_id,
        "method": method,
        "heldout_crop": "kiwi",
        "n_labels": actual_budget,
        "seed": seed,
        "mAP50": round(mAP50, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "model_size_mb": round(model_size_mb, 2),
        "train_time_min": round(train_time, 2),
        "checkpoint_path": str(best_ckpt.relative_to(ROOT)),
    }

    print(f"\n  ✓ mAP50={row['mAP50']}  P={row['precision']}  R={row['recall']}  "
          f"F1={row['f1']}  time={row['train_time_min']}min")

    # ── Cleanup temp data ──────────────────────────────────────────
    shutil.rmtree(tmp_dir)

    return row


SUMMARY_HEADER = [
    "result_id", "run_id", "method", "heldout_crop", "n_labels",
    "mAP50", "precision", "recall", "f1", "latency_ms",
    "model_size_mb", "train_time_min", "split_version", "seed",
]


def append_results(row: dict) -> None:
    """Append a result row to summary.csv (creating it with a header if absent)."""
    result_id = f"r_{row['run_id']}"
    new_file = not RESULTS_CSV.exists()
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(SUMMARY_HEADER)
        writer.writerow([
            result_id,
            row["run_id"],
            row["method"],
            row["heldout_crop"],
            row["n_labels"],
            row["mAP50"],
            row["precision"],
            row["recall"],
            row["f1"],
            "",  # latency_ms (not measured)
            row["model_size_mb"],
            row["train_time_min"],
            "v1",  # split_version
            row["seed"],
        ])


def append_registry(row: dict) -> None:
    """Append a run entry to registry.csv."""
    today = datetime.now().strftime("%Y-%m-%d")
    with open(REGISTRY_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            row["run_id"],
            today,
            row["method"],
            "kiwi",
            "kiwi",
            row["n_labels"],
            row["seed"],
            str(CONFIG_PATH.relative_to(ROOT)),
            f"python scripts/run_baselines.py --methods {row['method']} --seeds {row['seed']} --budgets {row['n_labels']}",
            row.get("checkpoint_path", ""),
            "completed",
            f"mAP50={row['mAP50']}",
        ])


def main():
    parser = argparse.ArgumentParser(description="Run Step 3 baselines on kiwi.")
    parser.add_argument(
        "--methods", nargs="+", default=["scratch", "coco_transfer"],
        choices=["scratch", "coco_transfer"],
        help="Baseline methods to run",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=DEFAULT_SEEDS,
        help=f"Random seeds (default: {DEFAULT_SEEDS})",
    )
    parser.add_argument(
        "--budgets", nargs="+", type=int, default=DEFAULT_BUDGETS,
        help=f"Label budgets (default: {DEFAULT_BUDGETS})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would run without training",
    )
    args = parser.parse_args()

    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load completed runs to skip reruns ─────────────────────────
    completed_ids = load_completed_run_ids()

    total = len(args.methods) * len(args.seeds) * len(args.budgets)
    done = 0
    skipped = 0
    print(f"Step 3 Baseline Evaluation")
    print(f"  Methods: {args.methods}")
    print(f"  Seeds:   {args.seeds}")
    print(f"  Budgets: {args.budgets}")
    print(f"  Device:  {DEVICE or 'auto'}")
    print(f"  Total runs: {total}")
    print(f"  Already completed: {len(completed_ids)}")

    for method in args.methods:
        for seed in args.seeds:
            for budget in args.budgets:
                run_id = f"{method}_kiwi_b{budget}_s{seed}"
                if run_id in completed_ids:
                    skipped += 1
                    done += 1
                    print(f"\n  ⏭ Skipping {run_id} (already in summary.csv) [{done}/{total}]")
                    continue
                row = run_single_experiment(method, seed, budget, dry_run=args.dry_run)
                if row:
                    append_results(row)
                    append_registry(row)
                done += 1
                print(f"\n  [{done}/{total}] complete")

    print(f"\n{'='*60}")
    print(f"  All {total} runs finished ({skipped} skipped, {total - skipped} trained).")
    if not args.dry_run:
        print(f"  Results → {RESULTS_CSV}")
        print(f"  Registry → {REGISTRY_CSV}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
