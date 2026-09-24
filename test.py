"""CLI smoke test for the supplied model and corrected Grad-CAM.

Example:
    python test.py --model ../attached_assets/model_1789918225387.pth \
        --image ../attached_assets/frame.jpg --output-dir test_output
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from dust_level import analyze_dust
from gradcam import apply_dust_heatmap, read_bgr
from model import load_model, predict_rgb


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("test_output"), type=Path)
    args = parser.parse_args()

    model = load_model(args.model)
    image_bgr = read_bgr(args.image)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    label, confidence, probabilities = predict_rgb(model, image_rgb)
    overlay_rgb, heatmap_rgb, coverage = apply_dust_heatmap(image_bgr, model)
    analysis = analyze_dust(confidence, coverage, label)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(
        str(args.output_dir / "overlay.png"),
        cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR),
    )
    cv2.imwrite(
        str(args.output_dir / "heatmap.png"),
        cv2.cvtColor(heatmap_rgb, cv2.COLOR_RGB2BGR),
    )

    print(f"Prediction : {label}")
    print(f"Confidence : {confidence:.2f}%")
    print(f"Probabilities: Clean={probabilities[0]:.4f}, Dusty={probabilities[1]:.4f}")
    print(f"Coverage   : {coverage:.2f}%")
    print(f"Dust level : {analysis.dust_level}")
    print(f"Overlay    : {args.output_dir / 'overlay.png'}")
    print(f"Heatmap    : {args.output_dir / 'heatmap.png'}")


if __name__ == "__main__":
    main()
