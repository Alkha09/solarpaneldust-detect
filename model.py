"""
model.py — Shared model architecture and loading logic
Used by: input_video.py, predict.py, heatmap.py, dashboard.py, app.py
"""

import os
import torch
import torch.nn as nn
from torchvision import models

# --------------------------------
# Device
# --------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Model device: {device}")

# --------------------------------
# Classes
# --------------------------------
CLASSES = ["Clean", "Dusty"]

# --------------------------------
# Build model — matches train_v2.py exactly
# --------------------------------
def build_model():
    """Build MobileNetV2 with custom classifier"""
    m = models.mobilenet_v2(weights=None)
    m.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(m.last_channel, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 2),
    )
    return m

def load_model(model_path="model.pth"):
    """
    Load trained model weights.
    
    Args:
        model_path: Path to the .pth file
    
    Returns:
        model: Loaded model in eval mode
    
    Raises:
        FileNotFoundError: If model file doesn't exist
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: '{model_path}'")
    
    model = build_model()
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.to(device)
    model.eval()
    print(f"✓ Model loaded from '{model_path}'")
    return model

def get_model_info(model):
    """Get model information for display"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "device": str(device),
        "total_params": total_params,
        "trainable_params": trainable_params,
        "classes": CLASSES
    }