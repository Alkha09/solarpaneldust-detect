"""Video inference and heatmap rendering for SolarSense."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch

from dust_level import analyze_dust
from gradcam import MAX_RENDER_SIDE, apply_dust_heatmap
from model import predict_rgb
from segmentation import predict_dust_mask, UNet


ProgressCallback = Callable[[int, dict], None]


def _annotate(
    frame_bgr: np.ndarray, label: str, confidence: float, coverage: float
) -> np.ndarray:
    frame = frame_bgr.copy()
    color = (50, 220, 80) if label == "Clean" else (60, 80, 240)
    cv2.rectangle(frame, (12, 12), (390, 112), (15, 23, 42), thickness=-1)
    cv2.putText(
        frame, f"Prediction: {label}", (25, 43),
        cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA,
    )
    cv2.putText(
        frame, f"Confidence: {confidence:.1f}%", (25, 72),
        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (235, 235, 235), 2, cv2.LINE_AA,
    )
    cv2.putText(
        frame, f"Panel dust: {coverage:.1f}%", (25, 99),
        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (235, 235, 235), 2, cv2.LINE_AA,
    )
    return frame


def process_video(
    input_path: str | Path,
    output_path: str | Path,
    model: torch.nn.Module,
    skip_frames: int = 12,
    show_heatmap: bool = False,
    progress_cb: ProgressCallback | None = None,
    segmentation_model: UNet | None = None,
) -> dict:
    """Process a video and return UI-ready summary metrics.

    The default policy intentionally limits the number of frames sent through the
    model so long videos finish in a practical time while still producing a valid
    summary. Heatmaps are expensive and are disabled by default unless the caller
    explicitly requests them.
    """
    if skip_frames < 0:
        raise ValueError("skip_frames must be zero or greater.")

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError("Video has invalid dimensions.")

    render_scale = min(1.0, MAX_RENDER_SIDE / float(max(width, height)))
    render_width = max(1, int(round(width * render_scale)))
    render_height = max(1, int(round(height * render_scale)))

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (render_width, render_height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create output video: {output}")

    sampled: list[dict] = []
    per_second: dict[int, list[dict]] = defaultdict(list)
    last_label = "Clean"
    last_confidence = 0.0
    last_coverage = 0.0
    representative_overlay_rgb: np.ndarray | None = None
    representative_heatmap_rgb: np.ndarray | None = None
    frame_index = 0

    target_sample_count = 12
    sample_stride = max(skip_frames + 1, max(1, total_frames // max(1, target_sample_count)))

    try:
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            if render_scale < 1.0:
                frame_bgr = cv2.resize(
                    frame_bgr,
                    (render_width, render_height),
                    interpolation=cv2.INTER_AREA,
                )

            should_process = frame_index % sample_stride == 0
            if should_process:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                last_label, last_confidence, _ = predict_rgb(model, frame_rgb)
                if show_heatmap:
                    overlay_rgb, heatmap_rgb, last_coverage = apply_dust_heatmap(
                        frame_bgr, model
                    )
                    representative_overlay_rgb = overlay_rgb
                    representative_heatmap_rgb = heatmap_rgb
                    rendered_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
                else:
                    last_coverage = 0.0
                    rendered_bgr = frame_bgr
                if segmentation_model is not None:
                    _mask, last_coverage = predict_dust_mask(segmentation_model, frame_bgr)

                analysis = analyze_dust(
                    last_confidence, last_coverage, last_label
                )
                sample = {
                    "frame": frame_index,
                    "second": round(frame_index / max(fps, 1.0), 2),
                    "label": last_label,
                    "confidence": round(last_confidence, 1),
                    "coverage": round(last_coverage, 1),
                    "dust_level": analysis.dust_level,
                }
                sampled.append(sample)
                per_second[int(frame_index / max(fps, 1.0))].append(sample)
            else:
                rendered_bgr = frame_bgr

            writer.write(_annotate(
                rendered_bgr, last_label, last_confidence, last_coverage
            ))

            frame_index += 1
            if progress_cb and total_frames:
                progress_cb(
                    int(frame_index / total_frames * 100),
                    {
                        "frame": frame_index,
                        "total": total_frames,
                        "label": last_label,
                        "confidence": round(last_confidence, 1),
                        "coverage": round(last_coverage, 1),
                        "dust_level": analyze_dust(
                            last_confidence, last_coverage, last_label
                        ).dust_level,
                    },
                )
    finally:
        capture.release()
        writer.release()

    if not sampled:
        raise RuntimeError("No readable frames were found in the video.")

    overlay_url = None
    heatmap_url = None
    if representative_overlay_rgb is not None and representative_heatmap_rgb is not None:
        result_dir = output.parent.parent
        overlay_name = f"video_overlay_{output.stem}.png"
        heatmap_name = f"video_heatmap_{output.stem}.png"
        cv2.imwrite(str(result_dir / overlay_name), cv2.cvtColor(representative_overlay_rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(result_dir / heatmap_name), cv2.cvtColor(representative_heatmap_rgb, cv2.COLOR_RGB2BGR))
        overlay_url = f"/results/{overlay_name}"
        heatmap_url = f"/results/{heatmap_name}"

    dusty_count = sum(sample["label"] == "Dusty" for sample in sampled)
    avg_confidence = float(np.mean([sample["confidence"] for sample in sampled]))
    avg_coverage = float(np.mean([sample["coverage"] for sample in sampled]))
    overall_label = "Dusty" if dusty_count / len(sampled) >= 0.5 else "Clean"
    overall_analysis = analyze_dust(100.0, avg_coverage, overall_label)

    second_summary = []
    for second, rows in sorted(per_second.items()):
        second_summary.append({
            "second": second,
            "coverage": round(float(np.mean([row["coverage"] for row in rows])), 1),
            "confidence": round(
                float(np.mean([row["confidence"] for row in rows])), 1
            ),
            "label": (
                "Dusty"
                if sum(row["label"] == "Dusty" for row in rows) >= len(rows) / 2
                else "Clean"
            ),
        })

    return {
        "overall_label": overall_label,
        "overall_level": overall_analysis.dust_level,
        "dusty_percent": round(dusty_count / len(sampled) * 100.0, 1),
        "avg_confidence": round(avg_confidence, 1),
        "avg_coverage": round(avg_coverage, 1),
        "coverage_source": "U-Net pixel segmentation" if segmentation_model is not None else "Grad-CAM estimate",
        "video_duration": round(frame_index / max(fps, 1.0), 1),
        "frames_processed": len(sampled),
        "overlay_img": overlay_url,
        "heatmap_img": heatmap_url,
        "per_second": second_summary,
    }
