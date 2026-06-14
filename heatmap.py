"""
heatmap.py — Grad-CAM visualisation for SolarScan AI

Usage:
    python heatmap.py --image path/to/image.jpg
    python heatmap.py --image path/to/image.jpg --save out.png --no-window
"""

import argparse
import os
import sys
import datetime

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from model import load_model, CLASSES, device

# --------------------------------
# Args
# --------------------------------
parser = argparse.ArgumentParser(description="Grad-CAM heatmap for SolarScan")
parser.add_argument("--image", required=True, help="Input image path")
parser.add_argument("--model", default="model.pth", help="Path to model.pth")
parser.add_argument("--save", default="", help="Save overlay to this path")
parser.add_argument("--no-window", action="store_true", help="Skip imshow")
parser.add_argument("--threshold", type=int, default=150, help="Dust detection threshold")
args = parser.parse_args()

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
# Load model
# --------------------------------
try:
    model = load_model(args.model)
except FileNotFoundError as e:
    sys.exit(f"Error: {e}")

# --------------------------------
# Grad-CAM hooks
# --------------------------------
_gradients = None
_activations = None

def _forward_hook(module, inp, out):
    global _activations
    _activations = out

def _backward_hook(module, grad_in, grad_out):
    global _gradients
    _gradients = grad_out[0]

target_layer = model.features[-1]
target_layer.register_forward_hook(_forward_hook)
target_layer.register_full_backward_hook(_backward_hook)

# --------------------------------
# Load image
# --------------------------------
if not os.path.exists(args.image):
    sys.exit(f"Error: image not found — '{args.image}'")

img = cv2.imread(args.image)
if img is None:
    sys.exit(f"Error: could not decode image — '{args.image}'")

rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
tensor = transform(rgb).unsqueeze(0).to(device)

# --------------------------------
# Forward pass
# --------------------------------
model.eval()
tensor.requires_grad_(False)

output = model(tensor)
prob = F.softmax(output, dim=1)
conf, pred = torch.max(prob, 1)
label = CLASSES[pred.item()]
confidence = conf.item() * 100

print(f"\nPrediction: {label} ({confidence:.1f}%)")

# --------------------------------
# Backward pass for Grad-CAM
# --------------------------------
model.zero_grad()
score = output[0, pred.item()]
score.backward()

# --------------------------------
# Compute heatmap
# --------------------------------
pooled_grads = torch.mean(_gradients.detach(), dim=[0, 2, 3])
acts = _activations[0].detach().clone()

for i in range(acts.shape[0]):
    acts[i] *= pooled_grads[i]

heatmap = torch.mean(acts, dim=0)
heatmap = F.relu(heatmap)

if heatmap.max() > 0:
    heatmap /= heatmap.max()

heatmap = heatmap.cpu().numpy()

# --------------------------------
# Resize + colour map
# --------------------------------
heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap_uint8 = np.uint8(255 * heatmap_resized)
heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

overlay = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)

# --------------------------------
# Dust region bounding boxes
# --------------------------------
if label == "Dusty":
    _, thresh = cv2.threshold(heatmap_uint8, args.threshold, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    box_count = 0
    for c in contours:
        if cv2.contourArea(c) < 300:
            continue
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(overlay, "Dust", (x, max(y - 5, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        box_count += 1
    print(f"Dust regions detected: {box_count}")
else:
    print("Panel is Clean — no dust regions marked.")

# --------------------------------
# Save and show
# --------------------------------
if args.save:
    cv2.imwrite(args.save, overlay)
    print(f"Saved → {args.save}")

if not args.no_window:
    cv2.imshow("Original", img)
    cv2.imshow("Grad-CAM Heatmap", heatmap_color)
    cv2.imshow("Overlay", overlay)
    print("Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()