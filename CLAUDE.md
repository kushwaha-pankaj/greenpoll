# GreenPoll / CrossPoll — Agent Onboarding

YOLOv8n flower detection across greenhouse crops. CrossPoll = joint multi-crop pretraining → few-shot fine-tune on a held-out crop (kiwi).

## State (2026-05-04)

- **Steps 1–3 complete.** Env, data, scratch + COCO baselines done. 30 result rows in `results/summary.csv`.
- **Step 4 in flight on Kaggle right now.** Headless run launched as Notebook Version #4 of `notebook for training`. Joint pretrain on apple+strawberry+tomato → CrossPoll fine-tune sweep on kiwi (15 runs). ETA 3–4 hr from launch.
- When it finishes: download `results.zip` from notebook Output tab → unzip in repo root → commit `results/summary.csv` + `experiments/registry.csv` (skip `runs/` weights, gitignored).

## Repo layout that matters

- [scripts/run_baselines.py](scripts/run_baselines.py) — main sweep runner. Methods: `scratch`, `coco_transfer`, `crosspoll`. Auto-creates `results/summary.csv` with header. Dedups via `run_id` already in summary.csv.
- [scripts/harmonize_labels.py](scripts/harmonize_labels.py) — builds `data/joint/` for pretraining. Symlinks images, rewrites labels with `CLASS_MAP`. **Assumption**: tomato classes 0/1/2 all → flower; strawberry drops ripe(1)/unripe(2). Edit `CLASS_MAP` if wrong.
- [scripts/pretrain_joint.py](scripts/pretrain_joint.py) — joint pretrain. Output: `runs/joint/pretrain/weights/best.pt`. Required for `--methods crosspoll`.
- [configs/experiments/crosspoll.yaml](configs/experiments/crosspoll.yaml) — single source of truth for hyperparams. `label_budgets: [10, 25, 50, 100, 181]` (181 = kiwi train cap).
- [results/summary.csv](results/summary.csv) — tracked despite `results/` being gitignored (one-line `!results/summary.csv` exception). 15 columns: `..., mAP50, mAP50_95, ...`.
- `runs/`, `data/`, `*.pt` — gitignored. Local only.

## Kaggle workflow

Dataset: `kushwahapankaj2058/greenpoll-crops` (private, ~2.3 GB). Mounts at `/kaggle/input/datasets/kushwahapankaj2058/greenpoll-crops/` (Kaggle nests it inconsistently — code already handles both paths).

GitHub: https://github.com/kushwaha-pankaj/greenpoll (public). Local Mac has the old `flower-detection-framework` remote preserved under that name; `origin` → greenpoll.

Env vars the scripts honor:
- `GREENPOLL_DEVICE=0` → CUDA index. On Kaggle always set to `"0"`. Locally falls back to MPS via YAML.

User has limited local disk (~3.8 GB free). Don't restore `data/` locally without checking. Prefer Kaggle for any compute.

## Hard-won gotchas

- Kiwi train has only **181 images** — budgets above 181 silently cap. Old config had `[200, 500]` which collapsed to 181 twice. Now fixed.
- `results/` is in `.gitignore` as `results/*` with `!results/summary.csv` exception. Use `git add results/summary.csv` (no `-f` needed).
- Ultralytics writes `labels.cache` next to labels — fails on read-only `/kaggle/input/`. Always copy data into `/kaggle/working/greenpoll/data/` first.
- Script's `lr0=LR_FINETUNE` (0.0001) is used for **all** methods including scratch — likely why scratch collapses at low budgets. Not a bug introduced this session; documented but unfixed (would require re-running Step 3).
- mAP50_95 column is empty for the 30 Step 3 rows. Backfill skipped because it requires restoring `data/` locally. Step 4 sweep populates it for the 15 new rows.
- **Kaggle pip install gotcha**: do NOT `pip install ultralytics` on a Kaggle GPU notebook. It re-resolves and replaces the preinstalled CUDA `torch-2.10.0+cu128` with a CPU wheel `torch-2.10.0+cpu`, killing GPU access. Just `import ultralytics` — Kaggle's image already has it. Same for pyyaml.

## Recent commits (this session)

- `8c1a947` Step 4: harmonize_labels + pretrain_joint + crosspoll method
- `2c021ad` log mAP50-95, fix budget cap [10,25,50,100,181]
- `6f2f652` add Step 3 baseline results (30 runs from Kaggle)
- `e9f58ca` self-heal summary.csv if missing
- `9e6bcb4` initial Kaggle prep (env override, ignore node_modules)

## What to ask the user before doing anything risky

- Don't `rm -rf` data/runs without confirming Kaggle/zip backup exists.
- Don't change `CLASS_MAP` in harmonize_labels.py without asking — affects what counts as a flower.
- Don't push to `flower-detection-framework` remote — that's the old project name, kept for archival.
