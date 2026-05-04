# Next Actions

> Updated: 2026-03-25

## Immediate (Step 3 — COCO Baseline Evaluation)

1. **Create baseline evaluation script** (`scripts/run_baselines.py`)
   - Baseline #1: Train YOLOv8n from scratch on kiwi
   - Baseline #2: Fine-tune COCO-pretrained `yolov8n.pt` on kiwi
   - Iterate over label_budgets × seeds, log to `results/summary.csv`

2. **Sanity check** — Run 1 seed (42), smallest budget (25) for both baselines

3. **Full run** — All 5 budgets × 3 seeds × 2 methods = 30 training runs

## Queued (Step 4)

4. Create label harmonization script (remap all crops → single "flower" class)
5. Implement single-crop transfer baseline (#3)
6. Implement CrossPoll joint pretraining (#4)
