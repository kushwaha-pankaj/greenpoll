# Architecture Decision Records (ADR)

| # | Date | Decision | Rationale |
|---|------|----------|-----------|
| 1 | 2026-03-25 | Use YOLOv8n (nano) for all experiments | 3–5 hr/week budget; nano is fast enough for 5 budgets × 3 seeds × 4 methods = 60 runs |
| 2 | 2026-03-25 | Hold out kiwi as unseen test crop | Smallest dataset (302 imgs), true few-shot evaluation without contamination |
| 3 | 2026-03-25 | Train crops: tomato + strawberry + apple | Diverse flower morphologies; combined ~16k images for joint pretraining |
| 4 | 2026-03-25 | Use Roboflow YOLO export format | Direct compatibility with ultralytics; no manual annotation conversion |
| 5 | 2026-03-25 | Fix manifests to match actual class counts | Tomato had 3 classes (not 1); strawberry has 3 (flower/ripe/unripe); validated with label checker |
| 6 | 2026-03-25 | Cap label_budget at available train images | Kiwi has 181 train images; budget=200/500 must be capped to 181 |
| 7 | 2026-03-25 | Defer class harmonization to Step 4 | Step 3 baselines use original per-crop classes; Step 4 joint training remaps all to single "flower" class |
| 8 | 2026-03-25 | Seeds [42, 43, 44] for reproducibility | 3 seeds gives reasonable variance estimate without excessive compute |
