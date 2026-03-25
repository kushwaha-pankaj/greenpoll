# GreenPoll

GreenPoll is an open-source research software project for greenhouse pollination robotics.

Version 1 focuses on a software-only pipeline:
- flower detection
- target prioritization
- path planning
- simulation and evaluation

## Status
Early scaffold.

## Planned modules
- `greenpoll.data`: dataset loading and validation
- `greenpoll.detect`: flower detection baselines
- `greenpoll.prioritize`: target ranking
- `greenpoll.plan`: route planning
- `greenpoll.sim`: greenhouse-row simulation
- `greenpoll.eval`: metrics
- `greenpoll.viz`: visual outputs
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
