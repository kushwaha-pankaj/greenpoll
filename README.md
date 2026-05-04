# GreenPoll

GreenPoll is an open-source research software project for greenhouse pollination robotics.

**CrossPoll** is the first study: a multi-crop transfer-learning framework for flower detection, optimized for fast adaptation to a new crop with few labels.

## Current Status

- **Step 1** ✅ Environment setup (Python 3.14, venv, ultralytics 8.4.26)
- **Step 2** ✅ Data curation — 4 crop datasets downloaded (16,613 images, 2.3 GB YOLO format)
- **Step 3** 🔄 COCO baseline evaluation — in progress

## Datasets

| Crop | Images | Classes | Role |
|------|--------|---------|------|
| Apple | 10,379 | 1 (flower) | Training |
| Strawberry | 5,419 | 3 (flower, ripe, unripe) | Training |
| Tomato | 513 | 3 (class0, class1, class2) | Training |
| Kiwi | 302 | 1 (flower) | Held-out evaluation |

## Modules
- `greenpoll.data`: dataset loading and validation
- `greenpoll.detect`: flower detection baselines
- `greenpoll.eval`: metrics
- `greenpoll.viz`: visual outputs
- `greenpoll.plan`: route planning (future)
- `greenpoll.prioritize`: target ranking (future)
- `greenpoll.sim`: greenhouse-row simulation (future)
- `greenpoll.cli`: command-line tools

## Install
```bash
pip install -e .[dev]
```

## Run tests
```bash
pytest
```

## Research rules
- No fabricated results
- No fake citations
- No claims beyond implemented and validated functionality

## Project operating structure

This repo is organized to preserve continuity across chat sessions and keep code + paper synchronized.

### Continuity and decisions
- `ops/DECISIONS.md` — one-line ADR-style decisions with rationale
- `ops/SESSION_HANDOFF.md` — current state, blockers, and exact next 3 actions
- `ops/NEXT_ACTIONS.md` — short execution queue for the next session

### Experiment tracking
- `configs/datasets/` — dataset manifests and split configs
- `configs/experiments/crosspoll.yaml` — locked experiment settings
- `experiments/registry.csv` — run registry (id, seed, config, checkpoint, status)
- `results/summary.csv` — final metrics table used in plots and paper

### Manuscript workflow (LaTeX)
- `paper/manuscript/paper.tex` — main entry point
- `paper/manuscript/references.bib` — bibliography
- `paper/manuscript/sections/` — section-wise writing files

### Learning support
- `learning/README.md` — project-specific terms, methods, algorithms, and metrics explained

### Session workflow (recommended)
1. Update `ops/NEXT_ACTIONS.md` before starting.
2. Run experiment and append one row to `experiments/registry.csv`.
3. Add final metrics row to `results/summary.csv`.
4. Update `ops/DECISIONS.md` if any protocol changes.
5. End session by refreshing `ops/SESSION_HANDOFF.md`.
