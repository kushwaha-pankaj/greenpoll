# Known Learning Gaps And Documentation Drift

This file separates implemented repo behavior from older notes or planned
architecture, so the mentor lessons do not teach non-existent APIs as fact.

## Present In Documentation, Not Present In This Checkout

- `greenpoll.data`: The README lists this as a dataset loading and validation
  module, and `learning/README.md` references
  `greenpoll.data.validate_yolo_labels`. There is no `src/greenpoll/data/`
  package in the current checkout.
- `configs/datasets/`: The README describes dataset manifests under this path,
  but the current `configs/` tree contains only
  `configs/experiments/crosspoll.yaml`.
- `paper/manuscript/`: The README mentions an older manuscript path. The active
  arXiv paper tree in this checkout is `paper/arxiv/`.

## Stale Or Historical Notes

- The README says Step 3 COCO baseline evaluation is in progress. The current
  workspace notes and `results/summary.csv` show that Step 3 baseline rows have
  already been produced, and Step 4 CrossPoll work has been added.
- `learning/README.md` is valuable as a project journal, but it records earlier
  project state. Treat it as history unless a statement is confirmed by the
  current source files.

## How Lessons Should Handle These Gaps

- Teach implemented behavior from the current files under `scripts/`,
  `src/greenpoll/`, `configs/experiments/`, `tests/`, and `paper/arxiv/`.
- When discussing README-listed modules that are not present, label them as
  future-facing architecture rather than implemented code.
- Before making any paper claim, trace the number to `results/summary.csv` and
  the generation path through `scripts/build_paper_assets.py`.
