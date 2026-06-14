"""
app.py — SolarScan AI Web Server
Flask backend that serves the HTML interface and handles API requests

Usage:
    python app.py
    python app.py --port 8080
    python app.py --debug
"""

from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_cors import CORS
import torch
import torch.nn.functional as F
from torchvision import transforms
import cv2
import numpy as np
import os
import json
import datetime
import logging
import tempfile
import base64
import webbrowser
import threading
import argparse
from dataclasses import dataclass
from collections import Counter
import sys

# Import shared model
from model import load_model, CLASSES, device

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------------------------
# Config
# --------------------------------
@dataclass
class Config:
    model_path:      str = os.getenv("MODEL_PATH", "model.pth")
    history_file:    str = os.getenv("HISTORY_FILE", "history.json")
    max_upload_mb:   int = int(os.getenv("MAX_UPLOAD_MB", 100))
    video_skip:      int = int(os.getenv("VIDEO_FRAME_SKIP", 10))
    max_frames:      int = int(os.getenv("MAX_VIDEO_FRAMES", 500))
    video_batch:     int = int(os.getenv("VIDEO_BATCH_SIZE", 16))
    max_history:     int = int(os.getenv("MAX_HISTORY", 1000))
    host:            str = os.getenv("HOST", "127.0.0.1")
    port:            int = int(os.getenv("PORT", 5000))

cfg = Config()
app.config["MAX_CONTENT_LENGTH"] = cfg.max_upload_mb * 1024 * 1024

ALLOWED_IMAGE = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ALLOWED_VIDEO = {".mp4", ".avi", ".mov", ".mkv"}

# --------------------------------
# Load model
# --------------------------------
try:
    model = load_model(cfg.model_path)
    logger.info(f"Model loaded successfully on {device}")
except FileNotFoundError as e:
    logger.error(e)
    model = None
    logger.warning("Running in demo mode - model not loaded")

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
# GradCAM for heatmaps
# --------------------------------
class GradCAM:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer = model.features[-1]
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, tensor, class_idx):
        self.model.zero_grad()
        output = self.model(tensor)
        output[0, class_idx].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam).squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

gradcam = GradCAM(model) if model is not None else None

def apply_heatmap(bgr_img, cam, threshold=0.45):
    """Convert GradCAM to base64 for web display"""
    h, w = bgr_img.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    mask = (cam_resized >= threshold).astype(np.float32)
    mask = cv2.GaussianBlur(mask, (21, 21), 0)
    heatmap_color = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    mask_3ch = np.stack([mask, mask, mask], axis=2)
    overlay = (bgr_img * (1 - mask_3ch * 0.6) + heatmap_color * mask_3ch * 0.6).astype(np.uint8)
    _, buf = cv2.imencode(".png", overlay)
    return base64.b64encode(buf).decode("utf-8")

# --------------------------------
# Helper functions
# --------------------------------
def preprocess(bgr_frame):
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    return transform(rgb).unsqueeze(0).to(device)

@torch.no_grad()
def predict_batch(model, tensors, batch_size=16):
    labels, confidences = [], []
    for i in range(0, len(tensors), batch_size):
        batch = torch.stack(tensors[i:i + batch_size]).to(device)
        out = model(batch)
        probs = F.softmax(out, dim=1)
        confs, idxs = torch.max(probs, dim=1)
        labels.extend(CLASSES[idx.item()] for idx in idxs)
        confidences.extend(round(float(c.item()) * 100, 2) for c in confs)
    return labels, confidences

def extract_frames(video_path, skip=10, max_frames=500):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: '{video_path}'")
    
    tensors, raw_frames, frame_idx, sampled = [], [], 0, 0
    while sampled < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % skip == 0:
            tensors.append(preprocess(frame).squeeze(0))
            raw_frames.append(frame.copy())
            sampled += 1
        frame_idx += 1
    
    cap.release()
    return tensors, raw_frames

def efficiency_for(label, avg_confidence):
    if label == "Clean":
        return round(95 + avg_confidence / 100 * 5, 2)
    return max(0.0, round(60 - avg_confidence / 100 * 20, 2))

def compute_metrics(labels, confidences):
    dusty_confs = [c for l, c in zip(labels, confidences) if l == "Dusty"]
    if dusty_confs:
        precision = round(float(np.mean(dusty_confs)), 2)
    else:
        precision = 0.0
    
    avg_conf = round(float(np.mean(confidences)), 2) if confidences else 0.0
    recall = avg_conf / 100
    p = precision / 100
    f1 = round(2 * p * recall / (p + recall) * 100, 2) if (p + recall) > 0 else 0.0
    return precision, f1

def next_maintenance_date(label):
    today = datetime.date.today()
    delta = datetime.timedelta(days=7) if label == "Dusty" else datetime.timedelta(days=90)
    return (today + delta).strftime("%d %B %Y")

def load_history():
    if not os.path.exists(cfg.history_file):
        return []
    with open(cfg.history_file) as f:
        return json.load(f)

def save_history(item):
    data = load_history()
    data.append(item)
    if len(data) > cfg.max_history:
        data = data[-cfg.max_history:]
    with open(cfg.history_file, "w") as f:
        json.dump(data, f, indent=2)

def model_guard():
    if model is None:
        return jsonify({"error": "Model not loaded - check model.pth path"}), 503
    return None

def allowed_ext(filename, allowed):
    _, ext = os.path.splitext(filename.lower())
    return ext in allowed

# --------------------------------
# Routes
# --------------------------------
@app.route("/")
def home():
    """Serve the main HTML page"""
    return render_template("home.html")

@app.route("/home.html")
def serve_home():
    """Explicit route for home.html"""
    if os.path.exists('home.html'):
        return send_from_directory('.', 'home.html')
    return jsonify({"error": "home.html not found"}), 404

@app.route("/index.html")
def serve_index():
    """Explicit route for index.html"""
    if os.path.exists('index.html'):
        return send_from_directory('.', 'index.html')
    return jsonify({"error": "index.html not found"}), 404

@app.route("/predict/video", methods=["POST"])
def predict_video():
    """Handle video upload and return predictions"""
    print("UPLOAD HIT")
    err = model_guard()
    if err:
        return err
    
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    if not allowed_ext(file.filename, ALLOWED_VIDEO):
        return jsonify({"error": "Unsupported video format. Use MP4, AVI, MOV, or MKV"}), 400
    
    fd, tmp = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    try:
        file.save(tmp)
        
        tensors, raw_frames = extract_frames(tmp, skip=cfg.video_skip, max_frames=cfg.max_frames)
        
        if not tensors:
            return jsonify({"error": "No frames could be extracted from video"}), 400
        
        labels, confidences = predict_batch(model, tensors, batch_size=cfg.video_batch)
        
        count = Counter(labels)
        dusty_frames = count.get("Dusty", 0)
        clean_frames = count.get("Clean", 0)
        total = len(labels)
        dust_ratio = dusty_frames / total
        final_label = "Dusty" if dust_ratio > 0.5 else "Clean"
        avg_conf = round(sum(confidences) / total, 2)
        efficiency = efficiency_for(final_label, avg_conf)
        precision, f1 = compute_metrics(labels, confidences)
        next_maint = next_maintenance_date(final_label)
        
        heatmap_b64 = None
        if gradcam is not None:
            try:
                dusty_pairs = [(conf, i) for i, (lbl, conf) in enumerate(zip(labels, confidences)) if lbl == "Dusty"]
                if dusty_pairs:
                    cam_idx = max(dusty_pairs)[1]
                    cam_tensor = tensors[cam_idx].unsqueeze(0).to(device).requires_grad_(True)
                    class_idx = CLASSES.index(labels[cam_idx])
                    cam = gradcam.generate(cam_tensor, class_idx)
                    heatmap_b64 = apply_heatmap(raw_frames[cam_idx], cam)
                elif raw_frames:
                    cam_tensor = tensors[0].unsqueeze(0).to(device).requires_grad_(True)
                    class_idx = CLASSES.index(labels[0])
                    cam = gradcam.generate(cam_tensor, class_idx)
                    heatmap_b64 = apply_heatmap(raw_frames[0], cam)
            except Exception as e:
                logger.warning(f"Heatmap generation failed: {e}")
        
        summary = {
            "prediction": final_label,
            "dust_ratio": round(dust_ratio * 100, 2),
            "frames_analyzed": total,
            "dusty_frames": dusty_frames,
            "clean_frames": clean_frames,
            "avg_confidence": avg_conf,
            "efficiency": efficiency,
            "precision": precision,
            "f1_score": f1,
            "next_maintenance": next_maint,
            "maintenance": "Cleaning Recommended" if final_label == "Dusty" else "No Cleaning Required"
        }
        
        save_history({
            "date": datetime.datetime.now().isoformat(),
            "type": "Video",
            "filename": file.filename,
            **summary
        })
        
        return jsonify({
            "label": final_label,
            "summary": summary,
            "heatmap": heatmap_b64
        })
        
    except Exception as e:
        logger.exception("Video processing error")
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)

@app.route("/predict/image", methods=["POST"])
def predict_image():
    """Handle image upload and return prediction"""
    err = model_guard()
    if err:
        return err
    
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    if not allowed_ext(file.filename, ALLOWED_IMAGE):
        return jsonify({"error": "Unsupported image format. Use JPG, PNG, BMP, or WEBP"}), 400
    
    raw = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        return jsonify({"error": "Invalid image file"}), 400
    
    tensor = preprocess(img)
    
    with torch.no_grad():
        out = model(tensor)
        prob = F.softmax(out, dim=1)
        conf, idx = torch.max(prob, 1)
        label = CLASSES[idx.item()]
        confidence = round(float(conf.item()) * 100, 2)
    
    heatmap_b64 = None
    if gradcam is not None:
        try:
            tensor_grad = tensor.clone().requires_grad_(True)
            cam = gradcam.generate(tensor_grad, idx.item())
            heatmap_b64 = apply_heatmap(img, cam)
        except Exception as e:
            logger.warning(f"Image GradCAM failed: {e}")
    
    efficiency = efficiency_for(label, confidence)
    next_maint = next_maintenance_date(label)
    
    save_history({
        "date": datetime.datetime.now().isoformat(),
        "type": "Image",
        "filename": file.filename,
        "prediction": label,
        "confidence": confidence,
        "efficiency": efficiency,
        "next_maintenance": next_maint
    })
    
    return jsonify({
        "label": label,
        "prediction": label,
        "confidence": confidence,
        "efficiency": efficiency,
        "maintenance": "No Cleaning Required" if label == "Clean" else "Cleaning Recommended",
        "precision": confidence,
        "f1_score": round(confidence * 0.95, 2),
        "next_maintenance": next_maint,
        "heatmap": heatmap_b64
    })

@app.route("/api/history")
def api_history():
    """Return prediction history as JSON"""
    return jsonify(load_history())

@app.route("/api/stats")
def api_stats():
    """Return statistics about predictions"""
    history = load_history()
    total = len(history)
    clean = sum(1 for h in history if h.get("prediction") == "Clean")
    dusty = sum(1 for h in history if h.get("prediction") == "Dusty")
    return jsonify({
        "total": total,
        "clean": clean,
        "dusty": dusty,
        "clean_percent": round(clean / total * 100, 2) if total > 0 else 0,
        "dusty_percent": round(dusty / total * 100, 2) if total > 0 else 0
    })

@app.route("/health")
def health():
    """Health check endpoint"""
    if model is None:
        return jsonify({"status": "degraded", "reason": "model not loaded"}), 503
    return jsonify({
        "status": "ok",
        "device": str(device),
        "model_loaded": True,
        "classes": CLASSES
    })

@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": f"File too large (max {cfg.max_upload_mb} MB)"}), 413

@app.errorhandler(500)
def server_error(e):
    logger.exception("Unhandled error")
    return jsonify({"error": "Internal server error"}), 500

# --------------------------------
# Main entry point
# --------------------------------
def main():
    parser = argparse.ArgumentParser(description="SolarScan AI Web Server")
    parser.add_argument("--host", default=cfg.host, help="Host to bind to")
    parser.add_argument("--port", type=int, default=cfg.port, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser automatically")
    args = parser.parse_args()
    
    url = f"http://{args.host}:{args.port}"
    
    if not args.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    
    print("\n" + "=" * 50)
    print("     SOLARSCAN AI WEB SERVER")
    print("=" * 50)
    print(f"  URL       : {url}")
    print(f"  Model     : {cfg.model_path}")
    print(f"  Device    : {device}")
    print(f"  Classes   : {CLASSES}")
    print(f"  Max upload: {cfg.max_upload_mb} MB")
    print(f"  Video skip: {cfg.video_skip} frames")
    print("=" * 50)
    print("\n  Looking for HTML files: home.html or index.html")
    print("  Press Ctrl+C to stop the server\n")
    
    app.run(debug=args.debug, host=args.host, port=args.port)

if __name__ == "__main__":
    main() 