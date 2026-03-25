# Learning Journal - CrossPoll Project

## Overview
Tracking key insights, decisions, and lessons learned throughout the CrossPoll research project implementation.

---

## Phase 1: Environment & Setup (✅ Completed)
**What we did:** Set up Python environment, CLI infrastructure, dataset manifests

**Key Insights:**
- YOLOv8n (nano) is appropriate choice for this project due to:
  - Time budget: 3-5 hrs/week requires efficiency
  - Dataset size: ~2.3 GB total (small enough for nano, large enough for meaningful transfer learning)
  - Deployment target: Edge robot inference
  - Stability: Ultralytics tooling is mature vs fragile meta-learning alternatives
- YOLO format (bounding boxes + class_ids) is standard for object detection

**Tools Selected:**
- Model: Ultralytics YOLOv8n (COCO pretrained baseline)
- Framework: PyTorch (implicit via Ultralytics)
- Config system: YAML-based manifests for dataset metadata + central experiment config

---

## Phase 2: Data Curation & Download (✅ Completed)
**What we did:** Downloaded 4 flower detection datasets via Roboflow, created metadata manifests

**Dataset Breakdown:**
| Crop | Images | Size | Classes | Role |
|------|--------|------|---------|------|
| Apple | 10,379 | 1.9 GB | 1 (flower) | Training |
| Strawberry | 5,419 | 329 MB | 3 (flower, ripe, unripe) | Training |
| Tomato | 513 | 23 MB | 3 (class0, class1, class2) | Training |
| Kiwi | 302 | 18 MB | 1 (flower) | Held-out evaluation |
| **Total** | **16,613** | **2.3 GB** | - | - |

**Key Insights:**
- Roboflow dataset exports don't always match project metadata declarations
- Must validate class counts (`nc` field) against actual `data.yaml` files after download
- Image counts varied from expectations (strawberry: 4000→5419, apple: 10000→10379)
- Held-out crop strategy (kiwi) essential for unbiased few-shot evaluation

---

## Phase 3: Pre-Flight Verification & Data Quality Fixes (✅ Completed)
**What we did:** Comprehensive validation before Step 3, discovered and fixed manifest inconsistencies

### Critical Discovery: Manifest-Data Mismatch

**Problem:** Tomato dataset manifest declared 1 class, but actual Roboflow export contained 3 classes
- Manifest: `num_classes: 1, classes: [flower]`
- Actual downloaded: `nc: 3` with labels using `class_id ∈ {0, 1, 2}`

**Detection Method:**
```bash
$ python -m greenpoll.data.validate_yolo_labels data/tomato 1
# Result: Only 2 valid labels, 511 errors
# Error: "class_id 1 out of range [0, 0]"
# Error: "class_id 2 out of range [0, 0]"
```

**Root Cause:** Roboflow dataset classes don't always match project-level metadata. The dataset may have multi-class hierarchies not reflected in declarations.

**Resolution Applied:**
- Updated all 4 manifests with actual class counts:
  - Tomato: `nc: 1 → 3`, added `num_classes: 3`, class names `[class0, class1, class2]`
  - Strawberry: `nc: 3`, added `num_classes: 3` (was already correct)
  - Apple: `nc: 1`, added `num_classes: 1` for clarity
  - Kiwi: `nc: 1`, added `num_classes: 1`, clarified `role: heldout`
- Added harmonization notes to all manifests explaining Step 4 plan

### Lessons Learned:
1. **Always validate manifests against actual downloaded data** — Never trust declarations without verification
2. **Use YOLO label validator early** — Catches class mismatches before training fails
3. **Cross-validate across files** — Check manifest vs downloaded `data.yaml` versus actual labels on disk
4. **Document discrepancies** — Add notes explaining why harmonization is needed for future phases

---

## Phase 4: Design Decisions Rationale

### Why YOLOv8n not other variants?
- **YOLOv8s/m/l/x**: Would be too slow for 3-5 hrs/week budget (larger models = longer training × 5 label budgets × 3 seeds = 150+ model trainings needed)
- **YOLOv5/7**: Older, less maintained
- **Meta learning (MAML, Prototypical Nets)**: Fragile, require careful hyperparameter tuning
- **YOLOv8n**: Sweet spot — small enough for quick training, large enough for transfer learning

### Why COCO pretraining baseline in Step 3?
- Zero-shot transfer from general object detection to flower detection
- Establishes baseline before cross-crop transfer (Step 4) and harmonization (Step 4)
- Provides comparison point for "how much does domain transfer help?"

### Why hold out kiwi crop?
- True few-shot evaluation (model never sees kiwi during training)
- Prevents contamination of cross-crop training with test crop
- Isolates label budget effect from cross-crop transfer learning
- Enables fair comparison: "Does training on [tomato, strawberry, apple] help for kiwi?"

### Why remap all crops to "flower" in Step 4?
- Strawberry has [flower, ripe, unripe] classes (don't care about ripeness, only flower location)
- Tomato has [class0, class1, class2] (semantically unclear, probably different flower types)
- Apple already single "flower" class
- Harmonizing to unified "flower" class allows joint training without class confusion
- Step 3 uses original classes (baseline); Step 4 uses unified class (joint training)

---

## Current Status (Pre-Step 3)
✅ Environment working  
✅ Datasets downloaded and validated (16,613 images, 2.3 GB, YOLO format)  
✅ Manifests corrected for actual class counts  
✅ Experiment config ready (crosspoll.yaml: yolov8n, seeds [42, 43, 44], label budgets [25,50,100,200,500])  
✅ Git commited with manifest fixes (commit 260763f)  

🔜 **Next: Step 3 - COCO Baseline Evaluation**
- Run YOLOv8n zero-shot on each crop
- Log mAP@50 metrics
- Establish performance baseline

---

## Notes for Future Steps

**Step 4 - Joint Training with Harmonization**
- Create `scripts/harmonize_labels.py` for cross-crop remapping
- Remap to unified "flower" class across all crops
- Joint train on [tomato, strawberry, apple]
- Fine-tune on kiwi with label budgets

**Step 5+ - Full CrossPoll Pipeline**
- Implement active learning loop
- Label selection based on model uncertainty + diversity
- Evaluate annotation efficiency gains vs baselines

---

## Technical References

**YOLO Label Format:**
```
<class_id> <x_center> <y_center> <width> <height>
```
Where coordinates are normalized [0, 1] relative to image dimensions.

**Validation Command:**
```bash
python -m greenpoll.data.validate_yolo_labels <dataset_path> <num_samples>
```

**Dataset Manifest Schema:**
```yaml
num_classes: int
classes: [class_name1, class_name2, ...]
image_count: int
role: training | heldout
notes: string
```
