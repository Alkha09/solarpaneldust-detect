import argparse
import os
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import cv2
import numpy as np
from collections import Counter

# ----------------------------
# Device
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ----------------------------
# Model
# ----------------------------
def build_model():
    model = models.mobilenet_v2(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.last_channel, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 2)
    )
    return model


def load_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")

    model = build_model()
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    print("Loaded model:", path)
    return model


# ----------------------------
# Transform
# ----------------------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

CLASSES = ["Clean", "Dusty"]

# ----------------------------
# GradCAM
# ----------------------------
class GradCAM:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None

        layer = model.features[-1]
        layer.register_forward_hook(self.forward_hook)
        layer.register_full_backward_hook(self.backward_hook)

    def forward_hook(self, module, inp, out):
        self.activations = out.detach()

    def backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, x, class_idx):
        self.model.zero_grad()
        out = self.model(x)
        out[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)

        cam = F.relu(cam).squeeze().cpu().numpy()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam


def save_heatmap(frame, cam, path):
    h, w = frame.shape[:2]
    cam = cv2.resize(cam, (w, h))

    heatmap = cv2.applyColorMap(np.uint8(cam * 255), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)

    cv2.imwrite(path, overlay)
    print("Heatmap saved:", path)


# ----------------------------
# Utils
# ----------------------------
def preprocess(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return transform(rgb)


@torch.no_grad()
def predict_batch(model, tensors, batch_size=16):
    labels, confidences = [], []

    for i in range(0, len(tensors), batch_size):
        batch = torch.stack(tensors[i:i+batch_size]).to(device)

        out = model(batch)
        probs = F.softmax(out, dim=1)

        conf, idx = torch.max(probs, dim=1)

        labels.extend(CLASSES[i.item()] for i in idx)
        confidences.extend([float(c.item()) * 100 for c in conf])

    return labels, confidences


def extract_frames(video_path, skip=3, max_frames=200):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError("Cannot open video")

    tensors, frames = [], []
    frame_id, sampled = 0, 0

    while sampled < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % skip == 0:
            tensors.append(preprocess(frame))
            frames.append(frame.copy())
            sampled += 1

        frame_id += 1

    cap.release()

    print("Frames sampled:", sampled)
    return tensors, frames


# ----------------------------
# Metrics
# ----------------------------
def efficiency(label, conf):
    if label == "Clean":
        return round(95 + conf / 100 * 5, 2)
    return round(max(0, 60 - conf / 100 * 20), 2)


def next_maintenance(label):
    today = datetime.date.today()
    days = 90 if label == "Clean" else 7
    return (today + datetime.timedelta(days=days)).strftime("%d %B %Y")


# ----------------------------
# MAIN
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--model", default="model.pth")
    parser.add_argument("--skip", type=int, default=3)
    parser.add_argument("--max", type=int, default=200)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--heatmap", default="heatmap.png")

    args = parser.parse_args()

    if not os.path.exists(args.video):
        raise FileNotFoundError("Video not found")

    model = load_model(args.model)
    cam = GradCAM(model)

    print("\nProcessing:", args.video)

    tensors, frames = extract_frames(
        args.video,
        skip=args.skip,
        max_frames=args.max
    )

    if not tensors:
        raise RuntimeError("No frames extracted")

    print("\nRunning inference...")
    labels, confidences = predict_batch(model, tensors, args.batch)

    # ----------------------------
    # Improved decision logic
    # ----------------------------
    count = Counter(labels)
    total = len(labels)

    dusty_frames = count.get("Dusty", 0)
    clean_frames = count.get("Clean", 0)

    dust_ratio = dusty_frames / total

    dust_score = sum(c for l, c in zip(labels, confidences) if l == "Dusty")
    clean_score = sum(c for l, c in zip(labels, confidences) if l == "Clean")

    final_label = "Dusty" if dust_score > clean_score else "Clean"

    avg_conf = sum(confidences) / total

    print("\n--- RESULT ---")
    print("Final:", final_label)
    print("Dust ratio:", round(dust_ratio * 100, 2), "%")
    print("Avg confidence:", round(avg_conf, 2))
    print("Efficiency:", efficiency(final_label, avg_conf))
    print("Next maintenance:", next_maintenance(final_label))

    if avg_conf < 60:
        print("WARNING: Low confidence → result may be unreliable")

    # ----------------------------
    # Heatmap
    # ----------------------------
    try:
        dusty_idx = [
            (c, i) for i, (l, c) in enumerate(zip(labels, confidences))
            if l == "Dusty"
        ]

        idx = max(dusty_idx)[1] if dusty_idx else 0

        x = tensors[idx].unsqueeze(0).to(device)
        cls = CLASSES.index(labels[idx])

        cam_img = cam.generate(x, cls)
        save_heatmap(frames[idx], cam_img, args.heatmap)

    except Exception as e:
        print("Heatmap failed:", e)


if __name__ == "__main__":
    main()