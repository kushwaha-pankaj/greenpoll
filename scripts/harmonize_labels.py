#!/usr/bin/env python3
"""Step 4 — Harmonize per-crop labels into a single 'flower' class.

For pretraining the CrossPoll multi-crop flower detector, all source-crop
labels are remapped to a single class id (0 = flower):

  apple:      class 0 (flower)              -> 0
  strawberry: class 0 (flower)              -> 0
              classes 1 (ripe), 2 (unripe)  -> dropped
  tomato:     classes 0, 1, 2               -> 0
              (Roboflow export uses generic class0/1/2 names; we assume
              all three are flower stages. Edit CLASS_MAP below to
              restrict if downstream inspection shows otherwise.)

Output goes to data/joint/<crop>/<split>/{images,labels}/ where
images is a symlink to the original images directory (no copy) and
labels are freshly written .txt files. Re-running this script is
idempotent — the joint dir is rebuilt from scratch.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
JOINT_DIR = DATA_DIR / "joint"

CLASS_MAP: dict[str, dict[int, int]] = {
    "apple":      {0: 0},
    "strawberry": {0: 0},
    "tomato":     {0: 0, 1: 0, 2: 0},
}

SPLITS = ["train", "valid"]


def harmonize_label_file(src: Path, dst: Path, mapping: dict[int, int]) -> int:
    """Rewrite one YOLO label file applying the class mapping. Returns kept count."""
    out_lines: list[str] = []
    with open(src) as f_in:
        for line in f_in:
            parts = line.strip().split()
            if not parts:
                continue
            cls = int(parts[0])
            if cls not in mapping:
                continue
            parts[0] = str(mapping[cls])
            out_lines.append(" ".join(parts))
    with open(dst, "w") as f_out:
        f_out.write("\n".join(out_lines))
        if out_lines:
            f_out.write("\n")
    return len(out_lines)


def main() -> None:
    if JOINT_DIR.exists():
        shutil.rmtree(JOINT_DIR)

    grand_files = grand_anns = 0
    for crop, mapping in CLASS_MAP.items():
        for split in SPLITS:
            src_img = DATA_DIR / crop / split / "images"
            src_lbl = DATA_DIR / crop / split / "labels"
            if not src_img.exists() or not src_lbl.exists():
                print(f"  skip {crop}/{split} (missing source)")
                continue

            dst_split = JOINT_DIR / crop / split
            dst_split.mkdir(parents=True, exist_ok=True)
            (dst_split / "labels").mkdir(exist_ok=True)
            # Copy (not symlink): Ultralytics resolves symlinks before deriving
            # label paths, which would bypass our harmonized labels and read
            # the originals. Copy keeps both image and label dirs co-located.
            shutil.copytree(src_img, dst_split / "images")

            n_files = n_anns = 0
            for lbl in src_lbl.glob("*.txt"):
                n_anns += harmonize_label_file(lbl, dst_split / "labels" / lbl.name, mapping)
                n_files += 1
            grand_files += n_files
            grand_anns += n_anns
            print(f"  {crop}/{split:5s}: {n_files:5d} files, {n_anns:6d} flower annotations")

    print(f"\nTotal: {grand_files} files, {grand_anns} flower annotations")
    print(f"Joint dataset at: {JOINT_DIR}")


if __name__ == "__main__":
    main()
