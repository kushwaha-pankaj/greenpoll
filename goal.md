# GreenPoll — MVP goal

## Single goal

Ship a **trustworthy, farmer-testable tool** that turns a **phone or laptop camera (or uploaded photos)** into **live flower detection**: **where** flowers are in the frame (bounding boxes), **how many**, and **how confident** the model is—so greenhouse operators can **see blooms quickly** without hand-counting every cluster.

The MVP proves the **perception layer** that later robotics or analytics can build on. It does **not** promise full **plant species identification**, **cultivar ID**, or **rich natural-language “analysis”** until those are separately scoped, labeled, and evaluated.

## How this connects to the research (CrossPoll)

The **paper track** studies **sample-efficient adaptation**: multi-crop **joint pretraining** (apple, strawberry, tomato) then **few-shot fine-tuning** on **held-out kiwi**, compared to **from-scratch** and **COCO transfer** training. The **MVP** reuses that stack’s **weights and honesty about limits**: users can run a **local demo** (`greenpoll-demo`) with a **kiwi-adapted `best.pt`** when available, or fall back to **joint pretrain** for pipeline demos—with **clear UI copy** so nobody confuses “flower detector” with “knows every crop.”

## Success criteria for MVP v1

1. **Functional:** User can load a checkpoint, feed an image or webcam frame, and get **overlay + count + confidence summary** in one session.
2. **Honest:** Copy and defaults explain **which weights** are loaded (kiwi fine-tune vs joint) and that **tuning** (e.g. confidence threshold) affects false positives vs misses.
3. **Traceable:** Public claims about **accuracy on kiwi** still come from the **evaluated protocol** (`results/summary.csv` / paper), not ad-hoc screenshots alone.
4. **Extendable later:** Architecture leaves room for **upload → label → fine-tune → gate → deploy** without pretending that flow exists fully in v1 unless implemented.

## Explicit non-goals for MVP v1

- Multi-tenant cloud training platform, accounts, or billing.
- Guaranteed correct **species** or **trait narrative** from a single detector head.
- “Works on any farm worldwide” without **new local evaluation**.

## Layered architecture (so the demo can grow without retraining the detector)

The MVP is structured as three independent layers. Each one can ship and improve on its own timeline. The **paper only describes Layer 1**; Layers 2 and 3 are product features that compose existing tools on top.

| Layer | What it answers | How it works | Status |
|---|---|---|---|
| **1 — Detection** | "Where are the flowers in this image and how many?" | The CrossPoll-trained YOLOv8n. Loaded by the demo from `checkpoints/joint_pretrain.pt` (joint fallback) or `runs/step3/.../weights/best.pt` (kiwi fine-tune). | **v1 — shipped** |
| **2 — Per-detection analysis** | "What is this flower? What growth stage? Any visible stress?" | Crop each detection box and call a vision-language model (Claude / GPT-4V / open-source VLM). Returns structured JSON. **No new training required**; the VLM already knows about plants. | **v2 — planned** |
| **3 — Aggregation** | "Field summary: 80% kiwi at full bloom, 20% wilting." | Pure Python post-processing over the per-detection JSONs. Counts, percentages, optional CSV export. | **v3 — planned** |

The wedge between Layer 1 and Layer 2 is deliberate. Layer 1 is a model with a measurable accuracy on a defined test set; we cite numbers from `results/summary.csv`. Layer 2 is generative; we hedge in the UI and never present its output as a measured metric.

## How this split protects both artifacts

- The **paper** stays focused on the one contribution it actually has evidence for (Layer 1, sample-efficient detection adaptation). Reviewers see a complete, well-scoped study.
- The **demo** stays free to grow. Adding stage classification (Layer 2) does not require re-running experiments, re-doing the paper, or collecting new annotated data. It's a swap-in module.
- When Layer 2 is added, the demo UI must clearly distinguish the two: "**Detected** 7 flowers" (from Layer 1, with a confidence number) versus "**The model thinks** these may be kiwi at the bud stage" (from Layer 2, hedged language, no claimed accuracy).

## One line for stakeholders

**GreenPoll MVP v1:** *See and count flowers in the greenhouse from the camera, with calibrated confidence—built on the CrossPoll research stack, scoped so we do not over-promise beyond detection. Future versions add per-flower description and field-level summaries as separate, composed layers.*
