"""Video inference and heatmap rendering for SolarSense."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import subprocess
from typing import Callable

import cv2
import imageio_ffmpeg
import numpy as np
import torch

try:
    from .dust_level import analyze_dust
    from .gradcam import MAX_RENDER_SIDE, apply_dust_heatmap
    from .model import predict_rgb
    from .segmentation import predict_dust_mask, UNet
except ImportError:  # pragma: no cover - script execution fallback
    from dust_level import analyze_dust
    from gradcam import MAX_RENDER_SIDE, apply_dust_heatmap
    from model import predict_rgb
    from segmentation import predict_dust_mask, UNet


ProgressCallback = Callable[[int, dict], None]


def _make_video_writer(output_path: Path, fps: float, width: int, height: int):
    """Create an OpenCV intermediate; final browser output is encoded separately."""
    codec = "mp4v"
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*codec),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(f"Could not open an MP4 intermediate writer for: {output_path}")
    return writer, codec


def _transcode_to_browser_mp4(intermediate_path: Path, output_path: Path) -> None:
    """Convert the OpenCV output to H.264, which browsers can decode in MP4."""
    temporary_path = output_path.with_name(f"{output_path.stem}.h264.tmp.mp4")
    temporary_path.unlink(missing_ok=True)
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(intermediate_path),
        "-an",
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(temporary_path),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            detail = result.stderr.strip()[-2000:] or "FFmpeg returned no diagnostic output."
            raise RuntimeError(f"H.264 video conversion failed: {detail}")
        if not temporary_path.exists() or temporary_path.stat().st_size <= 0:
            raise RuntimeError(
                f"H.264 video conversion produced no output file: {temporary_path}"
            )
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
        intermediate_path.unlink(missing_ok=True)


def _validate_written_video(output_path: Path, *, expected_frames: int, fps: float, width: int, height: int) -> dict:
    """Ensure the output file exists and is readable before exposing it to the UI."""
    if not output_path.exists():
        raise RuntimeError(f"Processed video file was not created: {output_path}")

    file_size = output_path.stat().st_size
    if file_size <= 0:
        raise RuntimeError(f"Processed video file is empty: {output_path} ({file_size} bytes)")

    capture = cv2.VideoCapture(str(output_path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Processed video file could not be opened for validation: {output_path}")

        actual_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        actual_fps = float(capture.get(cv2.CAP_PROP_FPS) or fps)

        if actual_frames <= 0 or actual_width <= 0 or actual_height <= 0:
            raise RuntimeError(
                "Processed video was written but has invalid media metadata. "
                f"Path: {output_path}, frames={actual_frames}, size=({actual_width}x{actual_height}), fps={actual_fps}"
            )
        if expected_frames > 0 and actual_frames != expected_frames:
            raise RuntimeError(
                "Processed video is incomplete: "
                f"expected {expected_frames} frames but found {actual_frames}."
            )
        ok, _frame = capture.read()
        if not ok:
            raise RuntimeError(f"Processed video has no decodable frames: {output_path}")

        return {
            "frames": actual_frames,
            "width": actual_width,
            "height": actual_height,
            "fps": actual_fps,
            "codec": "h264",
            "size_bytes": file_size,
        }
    finally:
        capture.release()


def summarize_video_samples(sampled: list[dict]) -> dict:
    """Aggregate sampled frames without letting a few noisy dusty frames flip the result.

    A single false-positive dusty frame should not force an otherwise clean panel
    video to appear dusty; the decision should reflect the average coverage and the
    share of dusty frames over the whole sample.
    """
    if not sampled:
        raise RuntimeError("No readable frames were found in the video.")

    coverage_values = [float(item.get("coverage", 0.0)) for item in sampled]
    confidence_values = [float(item.get("confidence", 0.0)) for item in sampled]
    avg_confidence = float(np.mean(confidence_values))
    avg_coverage = float(np.mean(coverage_values))
    dusty_ratio = sum(1 for item in sampled if item.get("label") == "Dusty") / len(sampled)
    dusty_percent = dusty_ratio * 100.0

    # Clean videos should stay clean unless dust is both persistent and materially
    # significant across the sampled frames.  This avoids false positives caused by
    # a single bright or shadowed frame in an otherwise clean panel sequence.
    if avg_coverage >= 18.0 and dusty_ratio >= 0.30:
        overall_label = "Dusty"
    elif avg_coverage >= 30.0 and avg_confidence >= 55.0:
        overall_label = "Dusty"
    elif dusty_ratio >= 0.60 and avg_coverage >= 10.0:
        overall_label = "Dusty"
    else:
        overall_label = "Clean"

    return {
        "overall_label": overall_label,
        "avg_confidence": round(avg_confidence, 1),
        "avg_coverage": round(avg_coverage, 1),
        "dusty_percent": round(dusty_percent, 1),
    }


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
    if output.exists():
        output.unlink(missing_ok=True)
    intermediate = output.with_name(f"{output.stem}.opencv.tmp.mp4")
    intermediate.unlink(missing_ok=True)

    try:
        writer, _writer_codec = _make_video_writer(
            intermediate, fps, render_width, render_height
        )
    except Exception as exc:  # pragma: no cover - depends on OpenCV codec availability
        capture.release()
        raise RuntimeError(
            f"Could not create intermediate output video: {intermediate}. {exc}"
        ) from exc

    sampled: list[dict] = []
    frames_written = 0
    per_second: dict[int, list[dict]] = defaultdict(list)
    last_label = "Clean"
    last_confidence = 0.0
    last_coverage = 0.0
    representative_overlay_rgb: np.ndarray | None = None
    representative_heatmap_rgb: np.ndarray | None = None
    frame_index = 0
    processing_succeeded = False

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
                    if last_label == "Clean":
                        last_coverage = 0.0
                    else:
                        last_coverage = max(
                            12.0,
                            min(100.0, max(0.0, (100.0 - last_confidence) * 1.5)),
                        )
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
            frames_written += 1

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
        processing_succeeded = True
    finally:
        capture.release()
        writer.release()
        if not processing_succeeded:
            intermediate.unlink(missing_ok=True)

    if frames_written == 0:
        intermediate.unlink(missing_ok=True)
        raise RuntimeError(
            f"No frames were written to the processed output video: {intermediate}"
        )
    if not sampled:
        intermediate.unlink(missing_ok=True)
        raise RuntimeError("No readable frames were found in the video.")

    _transcode_to_browser_mp4(intermediate, output)
    output_info = _validate_written_video(
        output,
        expected_frames=frames_written,
        fps=fps,
        width=render_width,
        height=render_height,
    )

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

    summary = summarize_video_samples(sampled)
    overall_label = summary["overall_label"]
    avg_confidence = summary["avg_confidence"]
    avg_coverage = summary["avg_coverage"]
    dusty_count = int(round(summary["dusty_percent"] / 100.0 * len(sampled)))
    overall_analysis = analyze_dust(avg_confidence, avg_coverage, overall_label)

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

    actual_total_frames = total_frames or frame_index
    return {
        "overall_label": overall_label,
        "overall_level": overall_analysis.dust_level,
        "dusty_percent": round(dusty_count / len(sampled) * 100.0, 1),
        "avg_confidence": round(avg_confidence, 1),
        "avg_coverage": round(avg_coverage, 1),
        "coverage_source": "U-Net pixel segmentation" if segmentation_model is not None else "Grad-CAM estimate",
        "video_duration": round(actual_total_frames / max(fps, 1.0), 1),
        "total_frames": actual_total_frames,
        "frames_processed": frame_index,
        "sampled_frames": len(sampled),
        "output_video_path": str(output),
        "output_video_size_bytes": output_info["size_bytes"],
        "output_video_width": output_info["width"],
        "output_video_height": output_info["height"],
        "output_video_fps": round(output_info["fps"], 2),
        "output_video_frames": output_info["frames"],
        "output_video_codec": output_info["codec"],
        "overlay_img": overlay_url,
        "heatmap_img": heatmap_url,
        "per_second": second_summary,
    }
