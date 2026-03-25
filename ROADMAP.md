# CrossPoll Project Roadmap

## Current Phase: Step 3 - COCO Baseline Evaluation

---

## Completed Phases ✅

### Step 1: Environment Setup
- [x] Python environment (venv) with PyTorch, YOLO, etc.
- [x] CLI infrastructure
- [x] Test suite for core modules
- [x] Git workflow
- **Status:** Working, all prerequisites met

### Step 2: Data Curation & Download
- [x] Create dataset manifests for 4 crops (tomato, strawberry, apple, kiwi)
- [x] Download datasets via Roboflow (~2.3 GB total, 16,613 images)
- [x] Convert to YOLO format (bounding boxes + class_ids)
- [x] Create train/val/test splits
- [x] Pre-flight validation and manifest correction
- **Status:** Complete, data validated and ready

---

## Current Phase: Step 3 - COCO Baseline

**Purpose:**  
Establish performance baseline using YOLOv8n pretrained on COCO dataset, without any additional training or domain transfer.

**Outputs:**
- mAP@50 scores for each crop on YOLO test split
- Baseline metrics documented in `results/summary.csv` and `experiments/registry.csv`

**Timeline:** ~2-3 hours (mostly PyTorch inference, not training)

**Deliverables:**
1. `scripts/eval_coco_baseline.py` — Script to run zero-shot inference
2. `results/baseline_metrics.json` — mAP@50 for each crop
3. Updated `results/summary.csv` with baseline row

**Key Metrics:**
```
mAP@50 (tomato)     = ?
mAP@50 (strawberry) = ?
mAP@50 (apple)      = ?
mAP@50 (kiwi)       = ? [evaluation only, not in training]
```

---

## Upcoming Phases 🔜

### Step 4: Single-Crop & Cross-Crop Transfer Learning

**Purpose:**  
Understand transfer learning gains from (a) single-crop, (b) cross-crop joint training, with label harmonization.

**Sub-steps:**
1. **Harmonize labels** — Remap all crops to single "flower" class
   - Tomato: [class0, class1, class2] → [flower]
   - Strawberry: [flower, ripe, unripe] → [flower]
   - Apple: [flower] → [flower] (no change)
   - Kiwi: [flower] → [flower] (no change)

2. **Single-crop baselines** — Train YOLOv8n on each crop separately
   - Tomato only, Strawberry only, Apple only
   - With varying label budgets: [25, 50, 100, 200, 500]
   - 3 seeds each for reproducibility

3. **Cross-crop joint training** — Train on combined [tomato + strawberry + apple]
   - Same label budgets and seeds
   - Evaluate on kiwi (held-out crop)

**Outputs:**
- Metrics comparing transfer vs single-crop performance
- Understanding of "Does training on multiple crops improve few-shot on held-out crop?"

**Timeline:** ~4-6 weeks (significant training time with label budgets and seeds)

---

### Step 5: Full CrossPoll Pipeline

**Purpose:**  
Active learning for efficient annotation — intelligently select which samples to label based on model uncertainty and diversity.

**Components:**
1. **Uncertainty sampling** — Query samples where model confidence is lowest
2. **Diversity sampling** — Select diverse samples (feature-space distance)
3. **Annotation loop** — Iteratively label, retrain, measure improvement
4. **Comparison** — CrossPoll vs random sampling vs uncertainty-only vs diversity-only

**Outputs:**
- Annotation efficiency curves (mAP vs number of labeled samples)
- Speedup factor vs random labeling

**Timeline:** ~2-3 weeks (implementation + evaluation)

---

## Experiment Config

**Location:** `configs/experiments/crosspoll.yaml`

```yaml
model: yolov8n.pt (COCO pretrained)
train_crops: [tomato, strawberry, apple]
heldout_crop: kiwi
label_budgets: [25, 50, 100, 200, 500]
seeds: [42, 43, 44]
primary_metric: mAP@50

dataset_paths:
  - configs/datasets/tomato.yaml
  - configs/datasets/strawberry.yaml
  - configs/datasets/apple.yaml
  - configs/datasets/kiwi.yaml
```

**Metrics Tracking:**
- Results: `results/summary.csv`
- Experiment registry: `experiments/registry.csv`
- Per-experiment details: `results/exp_<timestamp>.json`

---

## Dataset Reference

| Crop | Images | Size | Classes | Split | Role |
|------|--------|------|---------|-------|------|
| Apple | 10,379 | 1.9 GB | 1 | train/val/test | Training |
| Strawberry | 5,419 | 329 MB | 3 | train/val/test | Training |
| Tomato | 513 | 23 MB | 3 | train/val/test | Training |
| Kiwi | 302 | 18 MB | 1 | val/test | Held-out eval |
| **Total** | **16,613** | **2.3 GB** | - | - | - |

**Data Location:** `data/{tomato,strawberry,apple,kiwi}/` in YOLO format

---

## Key Design Decisions

### ✅ YOLOv8n Model Choice
- Efficiency: Fits 3-5 hrs/week budget
- Transfer learning: Strong COCO pretraining
- Deployment: Suitable for edge inference (robotics)
- Stability: Mature Ultralytics ecosystem

### ✅ Held-Out Crop Strategy
- Kiwi reserved for unbiased few-shot evaluation
- Prevents training-test contamination
- Enables fair cross-crop transfer comparison

### ✅ Label Budget Sweep
- [25, 50, 100, 200, 500] samples per crop
- Tests active learning efficiency across regimes
- Spans few-shot to modest-shot learning

### ✅ Three Seeds for Reproducibility
- [42, 43, 44] for statistical significance
- Allows error bars on reported metrics
- Mitigates random initialization effects

---

## Success Criteria

### Step 3 Complete When:
- ✅ COCO baseline mAP@50 measured for each crop
- ✅ Baseline metrics logged to `results/summary.csv`
- ✅ Reproducible with `scripts/eval_coco_baseline.py`

### Step 4 Complete When:
- ✅ Harmonized labels created and validated
- ✅ Single-crop baselines trained and evaluated
- ✅ Cross-crop joint training compared
- ✅ Analysis showing transfer learning benefits

### Step 5 Complete When:
- ✅ Active learning loop implemented
- ✅ Annotation efficiency curves generated
- ✅ CrossPoll vs baselines compared
- ✅ Paper-ready results and visualizations

---

## File Structure

```
learning/
├── README.md          ← Learning journal (this Phase)
└── CHEATSHEET.html    ← Quick reference

configs/
├── datasets/
│   ├── tomato.yaml
│   ├── strawberry.yaml
│   ├── apple.yaml
│   └── kiwi.yaml
└── experiments/
    └── crosspoll.yaml

scripts/
├── eval_coco_baseline.py     ← Step 3 (TBD)
├── harmonize_labels.py       ← Step 4 (TBD)
└── active_learning_loop.py   ← Step 5 (TBD)

results/
├── summary.csv              ← All experiments tracked here
└── exp_<timestamp>.json     ← Per-experiment details

data/
├── tomato/     → download/images/labels
├── strawberry/ → download/images/labels
├── apple/      → download/images/labels
└── kiwi/       → download/images/labels
```

---

## Time Budget

**Total Weekly:** 3-5 hours/week

**Breakdown (estimate):**
- Step 3: 1-2 hours / week × 2-3 weeks = 2-6 hours total ✅ CURRENT
- Step 4: 2-3 hours / week × 4-6 weeks (significant training overhead)
- Step 5: 2-3 hours / week × 2-3 weeks

---

## Next Actions

**Before Step 3:**
- [x] Verify all datasets downloaded and validated
- [x] Correct manifest inconsistencies (tomato class mismatch)
- [x] Commit data quality fixes to git
- [x] Update learning materials and roadmap ← YOU ARE HERE

**To Start Step 3:**
- [ ] Create `scripts/eval_coco_baseline.py`
- [ ] Run YOLOv8n zero-shot inference on test splits
- [ ] Log mAP@50 results to `results/summary.csv`
- [ ] Verify baseline metrics make sense (>0 mAP for all crops)

---

## Questions / Notes
- *Q: Why not train on all 4 crops together in Step 4?* 
  - A: Kiwi is held-out for unbiased evaluation; training on [tomato, strawberry, apple] only
  
- *Q: What if COCO baseline is already very high (>80 mAP)?*
  - A: Still valuable — shows transfer learning ceiling; active learning might have limited room to help
  
- *Q: When do we finalize the research paper?*
  - A: After Steps 3-5 complete, with results and plots integrated into LaTeX manuscript
