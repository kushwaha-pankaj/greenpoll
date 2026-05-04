# Session Handoff

> Updated: 2026-03-25

## Current State

- **Step 1** (Environment): ✅ Complete — Python 3.14, venv, greenpoll installed editable
- **Step 2** (Data): ✅ Complete — 4 datasets downloaded (16,613 images, 2.3 GB), manifests corrected
- **Step 3** (COCO Baselines): 🔄 In progress — deps installed, script not yet created

## Environment

- Python: 3.14.0 (`.venv/`)
- ultralytics: 8.4.26
- torch: 2.11.0
- MPS: Available (Apple Silicon)
- Command: `".venv/bin/python"`

## Blockers

None.

## Key Facts

- Kiwi train split: 181 images → budgets 200/500 must cap at 181
- Tomato/strawberry have 3 classes each; kiwi/apple have 1 class
- summary.csv and registry.csv have headers, no data rows yet
- `.gitignore` blocks `configs/` — use `git add -f` for config files

## Next 3 Actions

1. Create `scripts/run_baselines.py` — Baseline #1 (scratch) + #2 (COCO transfer) on kiwi
2. Sanity check with seed=42, budget=25
3. Full baseline run (30 training runs)
