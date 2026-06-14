"""
predict.py — single-image inference for SolarScan AI

Usage:
    python predict.py test_image.png
    python predict.py test_image.png --save out.png
    python predict.py test_image.png --json
"""

import argparse
import datetime
import json
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from model import load_model, CLASSES, device

# --------------------------------
# Transform
# --------------------------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# --------------------------------
# Inference
# --------------------------------
@torch.no_grad()
def predict(img_bgr: np.ndarray, model) -> dict:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tensor = transform(rgb).unsqueeze(0).to(device)

    probs = F.softmax(model(tensor), dim=1)
    conf, pred = torch.max(probs, 1)
    label = CLASSES[pred.item()]
    confidence = conf.item() * 100

    return {
        "label": label,
        "confidence": round(confidence, 2),
        "efficiency": _efficiency(label, confidence),
        "maintenance": _maintenance(label, confidence),
        "next_cleaning": str(_next_cleaning(label, confidence)),
    }

def _efficiency(label: str, confidence: float) -> float:
    if label == "Clean":
        return round(95 + (confidence / 100) * 5, 2)
    return max(0.0, round(60 - (confidence / 100) * 20, 2))

def _maintenance(label: str, confidence: float) -> str:
    if label == "Clean":
        return "No maintenance required"
    return "Immediate cleaning required" if confidence > 70 else "Cleaning recommended soon"

def _next_cleaning(label: str, confidence: float) -> datetime.date:
    today = datetime.date.today()
    if label == "Clean":
        return today + datetime.timedelta(days=15)
    days = 2 if confidence > 70 else 5
    return today + datetime.timedelta(days=days)

# --------------------------------
# Annotation
# --------------------------------
def annotate(img: np.ndarray, result: dict) -> np.ndarray:
    out = img.copy()
    y = 30
    cv2.putText(out, f"Prediction: {result['label']}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 80), 2)
    cv2.putText(out, f"Confidence: {result['confidence']:.2f}%", (10, y+30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 80, 0), 2)
    cv2.putText(out, f"Efficiency: {result['efficiency']}%", (10, y+60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 80, 200), 2)
    cv2.putText(out, f"Maintenance: {result['maintenance']}", (10, y+90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)
    cv2.putText(out, f"Next cleaning: {result['next_cleaning']}", (10, y+120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 2)
    return out

# --------------------------------
# Main
# --------------------------------
def main():
    parser = argparse.ArgumentParser(description="SolarScan single-image prediction")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--model", default="model.pth", help="Path to model.pth")
    parser.add_argument("--save", default="", help="Save annotated image")
    parser.add_argument("--no-window", action="store_true", help="Skip imshow")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        sys.exit(f"Error: image not found — '{args.image}'")

    img = cv2.imread(args.image)
    if img is None:
        sys.exit(f"Error: could not read image '{args.image}'")

    try:
        model = load_model(args.model)
    except FileNotFoundError as e:
        sys.exit(f"Error: {e}")

    result = predict(img, model)

    if args.json:
        print(json.dumps(result, indent=2))
        return

    print("=" * 40)
    for k, v in result.items():
        print(f"  {k:<14}: {v}")
    print("=" * 40)

    annotated = annotate(img, result)

    if args.save:
        cv2.imwrite(args.save, annotated)
        print(f"Saved → {args.save}")

    if not args.no_window:
        cv2.imshow("SolarScan — Prediction", annotated)
        print("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()