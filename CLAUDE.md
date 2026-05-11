# GreenPoll / CrossPoll — Agent Onboarding

YOLOv8n flower detection for greenhouse robotics. Two parallel tracks share one trained detector.

## Two tracks. Keep them separate.

This repo intentionally serves two different artifacts. Confusing them costs you both.

**Track A — The paper (`paper/arxiv/`)**
- Scope: CrossPoll only. Joint multi-crop pretraining → few-shot fine-tune on a held-out crop. Detection only.
- Audience: arXiv reviewers, future venue submissions.
- Numbers come from `results/summary.csv` via `scripts/build_paper_assets.py`. No hand-edited numbers in the paper.
- Limitations are stated honestly (single held-out crop, single architecture, etc.).
- **Do not extend the paper's claims to species ID, growth-stage classification, or "detailed analysis."** The paper has the data and evaluation only for detection. Anything beyond that goes in the demo, not the paper.

**Track B — The demo (`src/greenpoll/detect/gradio_app.py`, `goal.md`)**
- Scope: a browser-based product MVP that gives greenhouse operators "see and count flowers" today and grows toward richer analysis later.
- Audience: farmers, end users, conference / Twitter demo viewers, future stakeholders.
- Architecture is **layered** so each capability slots in independently (see `goal.md`):
  1. **Detection** — the CrossPoll-trained YOLO model. The only layer that needs labeled data + training.
  2. **Per-crop analysis** — vision-language model called on cropped detection boxes for "what is this and what stage". Composed from existing models, no new training required.
  3. **Aggregation** — Python post-processing that turns per-detection facts into a field-level summary.
- The demo can promise more than the paper because it composes ready-made tools. Keep UI copy honest about what the model knows vs. what the VLM is guessing.

**Rule of thumb when adding new code or content**:
- If it requires retraining the detector or changes evaluation numbers → it belongs to Track A. Update `paper/arxiv/`, `results/summary.csv`, `scripts/`.
- If it adds a downstream feature on top of detection → it belongs to Track B. Update `src/greenpoll/detect/gradio_app.py`, `goal.md`, README's "MVP demo" section.
- Cross-cutting concerns (env vars, the joint pretrain checkpoint, dataset access) live in scripts and configs that both tracks consume.

## Repo layout that matters

**Shared (used by both tracks)**
- [src/greenpoll/](src/greenpoll/) — installable Python package; `pip install -e .` exposes `greenpoll` and `greenpoll-demo`.
- [configs/experiments/crosspoll.yaml](configs/experiments/crosspoll.yaml) — single source of truth for training hyperparameters. `label_budgets: [10, 25, 50, 100, 181]` (181 = kiwi train cap).
- [checkpoints/joint_pretrain.pt](checkpoints/joint_pretrain.pt) — pinned 6 MB joint pretrain weights. Used by both `--methods crosspoll` in training and the demo's joint fallback.
- `runs/`, `data/`, `*.pt` (other than `checkpoints/*.pt`) — gitignored. Local only.

**Track A — paper assets**
- [paper/arxiv/](paper/arxiv/) — LaTeX preprint (sections, references.bib, results_table.tex, figures/sample_efficiency.pdf).
- [scripts/run_baselines.py](scripts/run_baselines.py) — main sweep runner. Methods: `scratch`, `coco_transfer`, `crosspoll`. Honors `--freeze N`, `--patience-override N`, `--lr0-override X`. Self-heals `summary.csv` if missing. Dedups via `run_id`.
- [scripts/harmonize_labels.py](scripts/harmonize_labels.py) — builds `data/joint/` for pretraining. Copies images, rewrites labels with `CLASS_MAP`. **Assumption**: tomato classes 0/1/2 all → flower; strawberry drops ripe(1)/unripe(2). Edit `CLASS_MAP` if wrong.
- [scripts/pretrain_joint.py](scripts/pretrain_joint.py) — joint pretrain. Output: `runs/joint/pretrain/weights/best.pt`. Required for `--methods crosspoll` (falls back to `checkpoints/joint_pretrain.pt` if local pretrain not present).
- [scripts/build_paper_assets.py](scripts/build_paper_assets.py) — reads `summary.csv`, writes `paper/arxiv/results_table.tex` + `paper/arxiv/figures/sample_efficiency.pdf`.
- [results/summary.csv](results/summary.csv) — tracked despite `results/` being gitignored (one-line `!results/summary.csv` exception). 15 columns.
- [experiments/registry.csv](experiments/registry.csv) — every fine-tune run, one row.

**Track B — demo assets**
- [goal.md](goal.md) — MVP scope, success criteria, non-goals, layered architecture.
- [src/greenpoll/detect/gradio_app.py](src/greenpoll/detect/gradio_app.py) — Gradio browser app. Run via `greenpoll-demo`.
- [tests/](tests/) — pytest for `greenpoll.plan` and `greenpoll.prioritize` modules.

## Demo build & run

```bash
pip install -e ".[demo]"      # adds gradio
greenpoll-demo                # default: auto-find newest kiwi fine-tune, fall back to joint pretrain
greenpoll-demo --crop joint   # force the joint pretrain
greenpoll-demo --weights path/to/best.pt --port 7861
```

Server binds to 127.0.0.1 by default. Use `--share` only when you intend a public temporary URL. UI distinguishes between `kiwi_finetune`, `joint_fallback`, `joint`, and `explicit` weight modes so users always know what they're seeing.

## Paper build

All numbers come from `results/summary.csv` via `scripts/build_paper_assets.py` — never edit the table or figure by hand.

```bash
# 1. (re)generate results_table.tex + figures/sample_efficiency.pdf from summary.csv
python scripts/build_paper_assets.py

# 2. compile (three options)
#    a) Tectonic single-binary engine — fastest local path, no sudo:
#       curl -fsSL https://drop-sh.fullyjustified.net | sh
#       ./tectonic -X compile paper/arxiv/main.tex
#    b) MacTeX/BasicTeX:
#       cd paper/arxiv && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
#    c) Overleaf — drag paper/arxiv/ into a new project, hit Recompile
```

Verification: no `[?]` citation markers, `wc -w paper/arxiv/sections/*.tex` ~ 4100 words. Style rules in the plan file at `/Users/PankajKushwaha/.claude/plans/cuddly-kindling-fairy.md` — vary sentence length, specific numbers always, hedging where appropriate, no AI-signature phrases (delve, leverage, moreover, furthermore, in conclusion).

If a reviewer asks for a number that isn't in `summary.csv`, run a new sweep, regenerate assets, then re-edit the prose. Don't add hand-calculated numbers.

## Kaggle workflow

Dataset: `kushwahapankaj2058/greenpoll-crops` (private, ~2.3 GB). Mounts at `/kaggle/input/datasets/kushwahapankaj2058/greenpoll-crops/` (Kaggle nests it inconsistently — code already handles both paths).

GitHub: https://github.com/kushwaha-pankaj/greenpoll (public). Local Mac has the old `flower-detection-framework` remote preserved under that name; `origin` → greenpoll.

Env vars the scripts honor:
- `GREENPOLL_DEVICE=0` → CUDA index. On Kaggle always set to `"0"`. Locally falls back to MPS via YAML.
- `GREENPOLL_DEMO_WEIGHTS=path/to/best.pt` → demo weight override (same effect as `--weights`).
- `GREENPOLL_ROOT=path/to/repo` → demo root override when default path resolution breaks.

User has limited local disk (~3.8 GB free). Don't restore `data/` locally without checking. Prefer Kaggle for any compute.

## Hard-won gotchas

- Kiwi train has only **181 images** — budgets above 181 silently cap.
- `results/` is gitignored as `results/*` with `!results/summary.csv` exception. Use `git add results/summary.csv` (no `-f` needed).
- Ultralytics writes `labels.cache` next to labels — fails on read-only `/kaggle/input/`. Always copy data into `/kaggle/working/greenpoll/data/` first.
- Script's `lr0=LR_FINETUNE` (0.0001) is used for **all** methods including scratch — likely why scratch collapses at low budgets. Documented limitation in the paper.
- mAP50_95 column is empty for several Step 3 baseline rows (logged before mAP50-95 was added). Step 4 rows are complete.
- **Kaggle pip install gotcha**: do NOT `pip install ultralytics` on a Kaggle GPU notebook. It re-resolves and replaces the preinstalled CUDA `torch-2.10.0+cu128` with a CPU wheel `torch-2.10.0+cpu`, killing GPU access. Use `pip install -q --no-deps ultralytics ultralytics-thop` instead.
- **CrossPoll b25 collapse**: in the first full Step 4 sweep, crosspoll fine-tune at b25 collapsed (mAP50 ≈ 0.39 across all seeds) due to small-batch noise destroying the joint pretrain's specialized features. Per-run logs show peak at epoch 3 then catastrophic forgetting until patience=15 stops it at epoch 18. **Fixed**: `--freeze 10` + `--patience-override 30` rescues the collapse. Run `python scripts/run_baselines.py --methods crosspoll --freeze 10 --patience-override 30` — produces distinct `crosspoll_freeze10_*` rows. Literature support: arxiv 2505.01016 (Gandhi & Gandhi) found freeze=10 best for similar fruit-detection task.

## What to ask the user before doing anything risky

- Don't `rm -rf` data/runs without confirming Kaggle/zip backup exists.
- Don't change `CLASS_MAP` in `harmonize_labels.py` without asking — affects what counts as a flower.
- Don't push to `flower-detection-framework` remote — that's the old project name, kept for archival.
- Don't extend the paper's claims to demo-only capabilities (species ID, growth stages, generated text). Two tracks, two scopes.
- Ignore everything under `learning/` when assessing paper or demo readiness — that's the user's personal study material, not part of the publishable artifact.
