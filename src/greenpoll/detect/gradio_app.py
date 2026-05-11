"""Gradio browser demo: flower detection from webcam or uploaded image.

Install optional deps: pip install -e ".[demo]"
Run: greenpoll-demo [--weights PATH]

Gradio is imported only inside main() so `import greenpoll` does not require gradio.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO


def joint_checkpoint_path(root: Path) -> Path:
    return root / "checkpoints" / "joint_pretrain.pt"


def find_newest_kiwi_finetune_best(root: Path) -> Path | None:
    """Newest ``runs/step3/**/weights/best.pt`` whose path mentions kiwi (run_baselines layout)."""
    base = root / "runs" / "step3"
    if not base.is_dir():
        return None
    candidates = [
        p
        for p in base.rglob("weights/best.pt")
        if p.is_file() and "kiwi" in p.as_posix().lower()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def resolve_demo_weights(
    root: Path,
    *,
    weights_cli: str | None,
    crop: str,
) -> tuple[Path, str]:
    """Return (checkpoint path, short label for UI: kiwi_finetune | joint | explicit)."""
    if weights_cli:
        return Path(weights_cli).expanduser().resolve(), "explicit"

    env_w = os.environ.get("GREENPOLL_DEMO_WEIGHTS")
    if env_w:
        return Path(env_w).expanduser().resolve(), "explicit"

    joint = joint_checkpoint_path(root)

    if crop == "joint":
        return joint, "joint"

    if crop == "kiwi":
        kiwi = find_newest_kiwi_finetune_best(root)
        if kiwi is None:
            raise SystemExit(
                "No kiwi fine-tuned checkpoint found under runs/step3/**/weights/best.pt.\n"
                "Copy a best.pt from your Kaggle run, or train locally, then retry.\n"
                "Or run with --crop joint (joint pretrain only), or pass --weights PATH."
            )
        return kiwi, "kiwi_finetune"

    # auto
    kiwi = find_newest_kiwi_finetune_best(root)
    if kiwi is not None:
        return kiwi, "kiwi_finetune"
    return joint, "joint_fallback"


def effective_repo_root() -> Path:
    """Resolve project root for default checkpoint paths.

    From ``.../repo/src/greenpoll/detect/gradio_app.py``, ``parents[0]`` is ``detect``,
    ``parents[3]`` is the repo root (``parents[n]`` matches ``.parent`` iterated ``n+1`` times).
    Override with ``GREENPOLL_ROOT`` when running from an unusual layout.
    """
    env = os.environ.get("GREENPOLL_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parents[3]


def resolve_predict_device(cli_device: str | None) -> int | str | None:
    """Match training script priority: CLI ``--device`` > ``GREENPOLL_DEVICE`` > auto."""
    if cli_device is not None and cli_device.strip() != "":
        s = cli_device.strip()
        return int(s) if s.isdigit() else s
    env = os.environ.get("GREENPOLL_DEVICE")
    if env:
        return int(env) if env.isdigit() else env
    return None


def build_stats_html(result: Any) -> str:
    """HTML summary card for one Ultralytics ``Results`` object."""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return (
            '<div class="gp-stats gp-stats--empty">'
            "<p class=\"gp-stats-title\">No flowers detected</p>"
            "<p class=\"gp-stats-hint\">Try a clearer photo, lower <code>--conf</code>, "
            "or a different checkpoint.</p></div>"
        )

    n = len(boxes)
    confs = boxes.conf
    mean_c = float(confs.mean().cpu()) if confs is not None and len(confs) else 0.0
    rows: list[str] = []
    names = getattr(result, "names", None) or {}
    if len(boxes) <= 20 and boxes.cls is not None:
        cls_ids = boxes.cls.cpu().int().tolist()
        conf_list = boxes.conf.cpu().tolist()
        for i, (cid, c) in enumerate(zip(cls_ids, conf_list, strict=False)):
            label = names.get(int(cid), str(int(cid))) if isinstance(names, dict) else str(int(cid))
            rows.append(
                f"<tr><td>{i + 1}</td><td><code>{label}</code></td>"
                f'<td class="gp-mono">{c:.3f}</td></tr>'
            )
    table = (
        '<table class="gp-table"><thead><tr><th>#</th><th>Class</th><th>Conf</th></tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table>"
    )
    return (
        '<div class="gp-stats">'
        '<div class="gp-stat-grid">'
        f'<div class="gp-stat-card"><span class="gp-stat-label">Detections</span>'
        f'<span class="gp-stat-value">{n}</span></div>'
        f'<div class="gp-stat-card"><span class="gp-stat-label">Mean confidence</span>'
        f'<span class="gp-stat-value gp-mono">{mean_c:.3f}</span></div>'
        "</div>"
        f"{table}</div>"
    )


def _infer_one(
    model: YOLO,
    image_rgb: Any,
    *,
    conf: float,
    imgsz: int,
    device: int | str | None,
) -> tuple[Any, str]:
    """Run YOLO once; return (annotated RGB ndarray, markdown stats)."""
    import numpy as np

    if image_rgb is None:
        return None, (
            '<div class="gp-stats gp-stats--empty"><p class="gp-stats-title">No image yet</p>'
            "<p class=\"gp-stats-hint\">Upload or capture a photo, then tap <strong>Run detection</strong>.</p></div>"
        )

    arr = np.asarray(image_rgb)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return None, (
            '<div class="gp-stats gp-stats--empty"><p class="gp-stats-title">Unsupported image</p>'
            "<p class=\"gp-stats-hint\">Expected RGB with shape (height, width, 3).</p></div>"
        )

    # Ultralytics / OpenCV pipelines expect BGR for raw ndarray in many paths.
    image_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    kwargs: dict[str, Any] = dict(conf=conf, imgsz=imgsz, verbose=False)
    if device is not None:
        kwargs["device"] = device

    results = model.predict(source=image_bgr, **kwargs)
    if not results:
        return None, (
            '<div class="gp-stats gp-stats--empty"><p class="gp-stats-title">Inference error</p>'
            "<p class=\"gp-stats-hint\">The model returned no results.</p></div>"
        )

    r0 = results[0]
    plotted_bgr = r0.plot()
    plotted_rgb = cv2.cvtColor(plotted_bgr, cv2.COLOR_BGR2RGB)
    stats = build_stats_html(r0)
    return plotted_rgb, stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GreenPoll flower detection demo (Gradio UI).",
    )
    p.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to a YOLO .pt checkpoint (overrides --crop and GREENPOLL_DEMO_WEIGHTS).",
    )
    p.add_argument(
        "--crop",
        choices=("auto", "kiwi", "joint"),
        default="auto",
        help=(
            "Which default weights to load if --weights is unset: "
            "'auto' = newest kiwi fine-tune under runs/step3/ if present, else joint pretrain; "
            "'kiwi' = require that kiwi checkpoint; 'joint' = checkpoints/joint_pretrain.pt only."
        ),
    )
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help='Force device (e.g. "0", "cpu", "mps"). Default: GREENPOLL_DEVICE env or auto.',
    )
    p.add_argument("--host", type=str, default="127.0.0.1", help="Bind address (default localhost)")
    p.add_argument("--port", type=int, default=7860, help="Port")
    p.add_argument(
        "--share",
        action="store_true",
        help="Create a temporary public Gradio link (off by default for safety).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = effective_repo_root()
    weights, weight_mode = resolve_demo_weights(root, weights_cli=args.weights, crop=args.crop)
    if not weights.is_file():
        raise SystemExit(f"Weights file not found: {weights}")

    try:
        import gradio as gr
    except ImportError as e:
        raise SystemExit(
            "Gradio is not installed. Run: pip install -e \".[demo]\""
        ) from e

    device = resolve_predict_device(args.device)
    model = YOLO(str(weights))

    if weight_mode == "kiwi_finetune":
        ckpt_note = (
            f"**Kiwi session:** using fine-tuned checkpoint\n`{weights}`\n\n"
            "**Upload a kiwi flower image** (or use the webcam), then click **Run detection**."
        )
    elif weight_mode == "joint_fallback":
        ckpt_note = (
            f"**Kiwi session (fallback):** no `runs/step3/.../weights/best.pt` found; using joint pretrain\n"
            f"`{weights}`\n\n"
            "You can still upload **kiwi** photos, but for paper-matched behavior copy a kiwi "
            "`best.pt` from Kaggle (or train locally), then restart with `--crop kiwi` or `--weights PATH`.\n\n"
            "**Upload a kiwi flower image** to try the pipeline, then click **Run detection**."
        )
    elif weight_mode == "joint":
        ckpt_note = (
            f"**Joint pretrain only:** `{weights}` (apple + strawberry + tomato flowers).\n\n"
            "**Upload a kiwi flower image** if you like, or any flower image; for kiwi-specific "
            "quality use `--crop auto` or `--weights` with a kiwi `best.pt`."
        )
    else:
        ckpt_note = (
            f"Loaded checkpoint: `{weights}`\n\n"
            "**Upload a kiwi flower image** (recommended for your experiment), then **Run detection**."
        )

    def infer(image: Any, progress: gr.Progress = gr.Progress()) -> tuple[Any, str]:
        progress(0.08, desc="Preparing image…")
        progress(0.2, desc="Running YOLO…")
        out, stats = _infer_one(model, image, conf=args.conf, imgsz=args.imgsz, device=device)
        progress(1.0, desc="Done")
        return out, stats

    demo_css = """
    .gp-wrap { max-width: 1180px; margin: 0 auto; }
    .gp-hero {
        text-align: center;
        padding: 0.5rem 0 1.25rem;
        border-bottom: 1px solid color-mix(in srgb, var(--border-color-primary) 70%, transparent);
        margin-bottom: 1.25rem;
    }
    .gp-hero h1 {
        font-size: 1.85rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        margin: 0 0 0.35rem;
        background: linear-gradient(120deg, var(--primary-500), var(--primary-700));
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
    }
    .gp-hero .gp-tagline { margin: 0; opacity: 0.78; font-size: 1rem; }
    .gp-steps {
        display: flex; gap: 1rem; flex-wrap: wrap; justify-content: center;
        margin: 0.75rem 0 0; padding: 0; list-style: none;
    }
    .gp-steps li {
        display: flex; align-items: center; gap: 0.5rem;
        padding: 0.35rem 0.85rem; border-radius: 999px;
        background: color-mix(in srgb, var(--primary-500) 12%, transparent);
        font-size: 0.9rem;
    }
    .gp-steps .gp-step-num {
        width: 1.35rem; height: 1.35rem; border-radius: 50%;
        background: var(--primary-500); color: var(--neutral-0);
        display: inline-flex; align-items: center; justify-content: center;
        font-size: 0.75rem; font-weight: 700;
    }
    .gp-meta {
        text-align: center; font-size: 0.85rem; opacity: 0.72;
        margin-top: 0.75rem;
    }
    .gp-meta code { font-size: 0.8rem; }
    .gp-panel label { font-weight: 600 !important; }
    .gp-stats {
        border-radius: 12px;
        padding: 1rem 1.1rem;
        background: color-mix(in srgb, var(--block-background-fill) 88%, var(--primary-500));
        border: 1px solid color-mix(in srgb, var(--border-color-primary) 60%, transparent);
    }
    .gp-stats--empty {
        text-align: center;
        background: var(--block-background-fill);
    }
    .gp-stats-title { font-weight: 600; margin: 0 0 0.35rem; font-size: 1.05rem; }
    .gp-stats-hint { margin: 0; opacity: 0.75; font-size: 0.9rem; line-height: 1.45; }
    .gp-stat-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.75rem;
        margin-bottom: 1rem;
    }
    .gp-stat-card {
        border-radius: 10px;
        padding: 0.75rem 1rem;
        background: var(--block-background-fill);
        border: 1px solid var(--border-color-primary);
        display: flex;
        flex-direction: column;
        gap: 0.2rem;
    }
    .gp-stat-label { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.06em; opacity: 0.65; }
    .gp-stat-value { font-size: 1.5rem; font-weight: 700; line-height: 1.1; color: var(--primary-600); }
    .gp-mono { font-variant-numeric: tabular-nums; font-family: ui-monospace, monospace; }
    .gp-table { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
    .gp-table th, .gp-table td { padding: 0.45rem 0.5rem; text-align: left; border-bottom: 1px solid var(--border-color-primary); }
    .gp-table th { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em; opacity: 0.65; }
    .gp-footer { text-align: center; margin-top: 1.5rem; padding-top: 1rem; font-size: 0.8rem; opacity: 0.55; }
    """

    try:
        theme = gr.themes.Soft(primary_hue="emerald", neutral_hue="slate", font="sans-serif")
    except (TypeError, ValueError):
        theme = gr.themes.Soft()

    with gr.Blocks(
        title="GreenPoll — flower detection",
        analytics_enabled=False,
        fill_width=True,
    ) as demo:
        with gr.Column(elem_classes=["gp-wrap"]):
            gr.HTML(
                "<div class=\"gp-hero\">"
                "<h1>GreenPoll</h1>"
                "<p class=\"gp-tagline\">Flower detection · research demo</p>"
                "<ol class=\"gp-steps\">"
                "<li><span class=\"gp-step-num\">1</span> Upload or capture</li>"
                "<li><span class=\"gp-step-num\">2</span> Run detection</li>"
                "</ol>"
                "<p class=\"gp-meta\">Device <code>"
                f"{device if device is not None else 'auto'}</code>"
                " · conf threshold <code>"
                f"{args.conf:g}</code></p>"
                "</div>"
            )

            with gr.Accordion("Session & checkpoint", open=False):
                gr.Markdown(ckpt_note)

            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    inp = gr.Image(
                        sources=["webcam", "upload"],
                        type="numpy",
                        label="Your image",
                        height=420,
                        elem_classes=["gp-panel"],
                    )
                with gr.Column(scale=1):
                    out = gr.Image(
                        type="numpy",
                        label="Detections overlay",
                        height=420,
                        elem_classes=["gp-panel"],
                    )

            run_btn = gr.Button(
                "Run detection",
                variant="primary",
                size="lg",
                scale=1,
            )
            stats = gr.HTML(
                '<div class="gp-stats gp-stats--empty">'
                '<p class="gp-stats-title">Ready</p>'
                "<p class=\"gp-stats-hint\">Add a photo above, then run detection to see counts and boxes.</p>"
                "</div>"
            )

            gr.HTML(
                '<p class="gp-footer">GreenPoll / CrossPoll · local demo · '
                "do not use <code>--share</code> unless you intend a public URL</p>"
            )

            run_btn.click(
                infer,
                inputs=inp,
                outputs=[out, stats],
                show_progress="full",
            )

    demo.queue(default_concurrency_limit=2)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
        theme=theme,
        css=demo_css,
        inbrowser=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
