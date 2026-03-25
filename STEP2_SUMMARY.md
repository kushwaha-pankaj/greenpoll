# Step 2 Data Infrastructure — Complete Setup

## Status
✅ **Complete** — All data curation components ready for use

## Components

### 1. Dataset Manifests (4 crops)
Located in `configs/datasets/`:
- **tomato.yaml** — 513 images, CC-BY-4.0
- **strawberry.yaml** — 4,000 images, 3 classes, CC-BY-4.0
- **apple.yaml** — 10,000 images, CC-BY-4.0, SAHI-tiled
- **kiwi.yaml** — 302 images, MIT, **held-out** for final eval only

### 2. Manifest Loader
`src/greenpoll/data/manifests.py`
- Load individual manifests: `load_manifest(name)`
- Load all manifests: `load_all()` → dict keyed by crop name
- List available manifests: `list_manifests()`
- CLI: `python -m greenpoll.data.manifests`

### 3. Download Automation
`scripts/download_datasets.py`
- Download all: `python scripts/download_datasets.py --all --api-key KEY`
- Download single crop: `python scripts/download_datasets.py --crop tomato --api-key KEY`
- Environment variable: `export ROBOFLOW_API_KEY="key"; python scripts/download_datasets.py --all`
- Interactive setup: `bash scripts/setup_roboflow_auth.sh`

### 4. YOLO Label Validator
`src/greenpoll/data/validate_yolo_labels.py`
- Per-file validation: `validate_yolo_label_file(path, num_classes)`
- Dataset validation: `validate_dataset(path, num_classes)`
- CLI: `python -m greenpoll.data.validate_yolo_labels data/tomato 1`

### 5. Documentation
- **data/README.md** — Dataset structure, download instructions, validation examples
- **DATASET_DOWNLOAD.md** — Complete download guide with troubleshooting
- **scripts/setup_roboflow_auth.sh** — Interactive authentication setup

## Quick Start Workflow

```bash
# 1. Set up authentication (one-time)
bash scripts/setup_roboflow_auth.sh

# 2. Download all datasets (~15GB)
python scripts/download_datasets.py --all

# 3. Validate labels
python -m greenpoll.data.validate_yolo_labels data/tomato 1
python -m greenpoll.data.validate_yolo_labels data/strawberry 3
python -m greenpoll.data.validate_yolo_labels data/apple 1
python -m greenpoll.data.validate_yolo_labels data/kiwi 1  # validation only, DO NOT TRAIN

# 4. Load manifests programmatically
python -c "from greenpoll.data.manifests import load_all; datasets = load_all(); print(datasets.keys())"
```

## Key Design Decisions

### Kiwi Freeze Policy
- **Never train** on kiwi until Step 5 (final evaluation)
- Kiwi is the target crop for cross-crop transfer learning
- Explicitly marked in manifest: `role: heldout`
- Documented in data/README.md with 3 emphatic warnings

### YOLO Format Validation
- Checks class_id ∈ [0, num_classes)
- Checks coordinates ∈ [0, 1]
- Per-split (train/val/test) error aggregation
- Helpful error messages with line numbers

### Roboflow Integration
- Uses official `roboflow` CLI package
- Maintains version 1 of each dataset for reproducibility
- API key passed via environment variable (secure, not in code)
- Clear error messages + troubleshooting guide

## Files Modified/Created This Session

**Commits:**
1. `d053cee` — Initial Step 2 helpers (downloader, validator, README)
2. `227f303` — Registry audit trail update
3. `eefe6bb` — Download script CLI syntax fix (positional dataset URL)
4. `87a56bf` — DATASET_DOWNLOAD.md creation
5. `e1799b4` — API key handling improvements + setup script
6. `0d922c0` — Troubleshooting guide added

**New Files:**
- `scripts/download_datasets.py`
- `scripts/setup_roboflow_auth.sh`
- `src/greenpoll/data/validate_yolo_labels.py`
- `data/README.md`
- `DATASET_DOWNLOAD.md`

**Updated Files:**
- `experiments/registry.csv` — 3 audit trail entries for Step 2 helpers

## Next Steps (Step 3)

Implement `scripts/eval_coco_baseline.py`:
1. Run YOLOv8n (COCO-pretrained) zero-shot on tomato/strawberry/apple/kiwi
2. Log mAP50 per crop to registry.csv
3. Log results to results/summary.csv
4. Compare pretraining disparities across crops

**Constraint:** Kiwi metrics are **validation only** — not used for training until Step 5 few-shot transfer.

## Audit Trail

Registry entries logged in `experiments/registry.csv`:
```
dataset_tomato,2026-03-25,manifest_add,513,1,CC BY 4.0,configs/datasets/tomato.yaml,added,Roboflow universe dataset
dataset_strawberry,2026-03-25,manifest_add,4000,3,CC BY 4.0,configs/datasets/strawberry.yaml,added,Roboflow universe dataset
dataset_apple,2026-03-25,manifest_add,10000,1,CC BY 4.0,configs/datasets/apple.yaml,added,Roboflow universe dataset
dataset_kiwi,2026-03-25,manifest_add,302,1,MIT,configs/datasets/kiwi.yaml,added,Held-out target crop
step2_downloader,2026-03-25,script_add,,,,,scripts/download_datasets.py,added,Roboflow dataset download automation
step2_validator,2026-03-25,script_add,,,,,src/greenpoll/data/validate_yolo_labels.py,added,YOLO label format validator
step2_data_docs,2026-03-25,doc_add,,,,,data/README.md,added,Dataset structure and freeze policy documentation
```

All components tested and verified. Ready for production use.
