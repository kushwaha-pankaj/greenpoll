# Step 2 Datasets - Download Complete

**Status**: ✅ ALL DATASETS DOWNLOADED AND VERIFIED  
**Date**: 2026-03-25
**Total Size**: ~16 GB

## Dataset Summary

| Crop | Total Images | Train | Val | Test | Classes | License | Status |
|------|-------------|-------|-----|------|---------|---------|--------|
| **Tomato** | 513 | 359 | 103 | 51 | 3* | CC BY 4.0 | ✓ Complete |
| **Strawberry** | 5,419 | 4,953 | 339 | 127 | 3 | CC BY 4.0 | ✓ Complete |
| **Apple** | 10,379 | 7,223 | 2,094 | 1,062 | 1 | CC BY 4.0 | ✓ Complete |
| **Kiwi** | 302 | 181 | 60 | 61 | 1 | MIT | ✓ Complete (HELD-OUT) |

*Tomato has 3 classes in data.yaml (metadata included from Roboflow; actual flower class is 0)

## YOLO Format Validation

✓ All datasets follow YOLO format with proper structure:
- `train/images/` and `train/labels/` (YOLO .txt format)
- `valid/images/` and `valid/labels/`
- `test/images/` and `test/labels/`

✓ Label files contain valid 5-field YOLO format:
- class_id (float)
- center_x (normalized 0-1)
- center_y (normalized 0-1)
- width (normalized 0-1)
- height (normalized 0-1)

## Download Command Used

```bash
python scripts/download_datasets.py --all --api-key B7HELQcvH1YmBkKJqBXK
```

## Directory Structure

```
data/
├── tomato/
│   ├── train/
│   ├── valid/
│   ├── test/
│   └── data.yaml
├── strawberry/
│   ├── train/
│   ├── valid/
│   ├── test/
│   └── data.yaml
├── apple/
│   ├── train/
│   ├── valid/
│   ├── test/
│   └── data.yaml
└── kiwi/
    ├── train/
    ├── valid/
    ├── test/
    └── data.yaml
```

## Next Steps

1. **Step 3**: Run COCO baseline evaluation
   - Evaluate YOLOv8n (COCO-pretraining) zero-shot on each crop
   - Log mAP50 metrics to `results/summary.csv`
   - Kiwi evaluation is **validation-only** (no training data)

2. **Usage Examples**:
   ```bash
   # Validate YOLO labels
   python -m greenpoll.data.validate_yolo_labels data/tomato 1
   
   # Load manifests programmatically
   from greenpoll.data.manifests import load_all
   datasets = load_all()
   ```

## Important Notes

⚠️ **Kiwi Freeze Policy**: Kiwi dataset is held-out for final evaluation in Step 5.
- DO NOT use kiwi for training until Step 5
- Kiwi metrics in Step 3 are validation-only
- Kiwi is the target crop for cross-crop transfer learning evaluation

## Verification Command

```bash
# Check all datasets exist
ls -lh data/tomato data/strawberry data/apple data/kiwi

# Count images per crop
for crop in tomato strawberry apple kiwi; do
  total=$(find data/$crop -name "*.jpg" -o -name "*.png" | wc -l)
  echo "$crop: $total images"
done
```

---

**Download Infrastructure**: All scripts (`scripts/download_datasets.py`, `scripts/setup_roboflow_auth.sh`), manifests (`configs/datasets/*.yaml`), and validators (`src/greenpoll/data/validate_yolo_labels.py`) are production-ready.

See [DATASET_DOWNLOAD.md](DATASET_DOWNLOAD.md) for troubleshooting and [STEP2_SUMMARY.md](STEP2_SUMMARY.md) for complete Step 2 reference.
