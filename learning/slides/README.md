# GreenPoll HTML Lessons

This folder contains browser-based slide lessons for learning the GreenPoll
codebase from first principles.

## How to View

Open any `lesson*.html` file in a browser. Each deck uses Reveal.js from a CDN,
so an internet connection is useful for the presentation styling.

Navigation:

- Right arrow or space: next slide.
- Left arrow: previous slide.
- Esc: overview mode.
- `?`: Reveal.js keyboard help.

## Lesson Contract

Every lesson follows the same teaching structure:

1. First-principles concept.
2. Application to this repository.
3. Why the code is written that way.
4. Key takeaways.
5. Questions and exercises.

When a slide mentions a paper number, result, or method comparison, it should
trace back to `results/summary.csv` or to a script that reads from it. Do not
teach hand-computed results as authoritative.

## Current Lesson Index

- `lesson00-repo-map.html`: project map, research integrity, and end-to-end flow.
- `lesson01-pyproject-testing.html`: packaging, imports, CI, and tests.
- `lesson06-harmonize-labels.html`: YOLO labels and class harmonization.
- `lesson07-pretrain-joint.html`: CrossPoll joint pretraining.
- `lesson08-run-baselines.html`: scratch, COCO transfer, and CrossPoll fine-tuning.
- `lesson09-paper-assets.html`: `summary.csv` to paper table and figure.

Use `_template.html` when adding future lessons.
