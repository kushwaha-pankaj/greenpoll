# Figures Directory

Add publication-ready figures here in PNG or PDF format.

## Required Figures

1. **pipeline.png** - CrossPoll pipeline diagram showing:
   - Stage 1: Joint pretraining on Apple, Strawberry, Tomato
   - Stage 2: Fine-tuning on Kiwi with varying label budgets
   - Stage 3: Evaluation against baselines
   
   Currently referenced in `sections/methods.tex` at the CrossPoll Pipeline section.

## Optional Figures

- Sample curves (mAP@50 vs label budget for all 4 methods)
- Feature visualization (t-SNE of learned embeddings)
- Sample detections (qualitative results on Kiwi flowers)
- Confusion matrices or error analysis

## Naming Convention

Use descriptive filenames:
- `pipeline.png` (required)
- `sample-efficiency-curves.png`
- `qualitative-results.png`
- etc.

## Image Quality

For arXiv submission:
- Minimum 300 DPI for publication
- PNG or PDF format preferred
- Width: 600-800px for best rendering in PDF
