"""Shared PyTorch model and preprocessing utilities for SolarSense.

The supplied checkpoint is a MobileNetV2 classifier with this head:
    Dropout(0.4) -> Linear(1280, 256) -> ReLU ->
    Dropout(0.3) -> Linear(256, 2)

Keep this file as the single source of truth for model construction and
preprocessing.  The previous project mixed this checkpoint with a TensorFlow
EfficientNet model and with a second, incompatible MobileNet head.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models


CLASS_NAMES = ("Clean", "Dusty")
IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def build_model(pretrained: bool = False) -> nn.Module:
    """Build the MobileNetV2 classifier, optionally with ImageNet features."""
    weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v2(weights=weights)
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.last_channel, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(256, len(CLASS_NAMES)),
    )
    return model


def _state_dict_from_checkpoint(checkpoint: Any) -> dict[str, torch.Tensor]:
    """Accept a raw state dict or a common checkpoint wrapper."""
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            candidate = checkpoint.get(key)
            if isinstance(candidate, dict):
                checkpoint = candidate
                break

    if not isinstance(checkpoint, dict):
        raise ValueError("The checkpoint does not contain a PyTorch state dict.")

    state_dict = dict(checkpoint)
    # DataParallel checkpoints often prefix every key with "module.".
    if state_dict and all(key.startswith("module.") for key in state_dict):
        state_dict = {key[7:]: value for key, value in state_dict.items()}
    return state_dict


def load_model(model_path: str | Path) -> nn.Module:
    """Load the supplied MobileNetV2 checkpoint in evaluation mode."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")

    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    state_dict = _state_dict_from_checkpoint(checkpoint)
    model = build_model()
    model.load_state_dict(state_dict, strict=True)
    model.to(DEVICE)
    model.eval()
    return model


def preprocess_rgb(image_rgb: np.ndarray | Image.Image) -> torch.Tensor:
    """Convert an RGB image to the exact tensor format used during training."""
    if isinstance(image_rgb, Image.Image):
        pil_image = image_rgb.convert("RGB")
    else:
        array = np.asarray(image_rgb)
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError("Expected an RGB image with shape (H, W, 3).")
        pil_image = Image.fromarray(array.astype(np.uint8), mode="RGB")

    pil_image = pil_image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    array = np.asarray(pil_image, dtype=np.float32) / 255.0
    array = (array - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(array.transpose(2, 0, 1)).float()
    return tensor.unsqueeze(0).to(DEVICE)


def predict_tensor(
    model: nn.Module, image_tensor: torch.Tensor
) -> tuple[int, float, np.ndarray]:
    """Return predicted class index, confidence in percent, and probabilities."""
    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    predicted_index = int(torch.argmax(probabilities).item())
    confidence = float(probabilities[predicted_index].item() * 100.0)
    return predicted_index, confidence, probabilities.detach().cpu().numpy()


def predict_rgb(
    model: nn.Module, image_rgb: np.ndarray
) -> tuple[str, float, np.ndarray]:
    """Predict an RGB image and return label, confidence percent, probabilities."""
    tensor = preprocess_rgb(image_rgb)
    index, confidence, probabilities = predict_tensor(model, tensor)
    return CLASS_NAMES[index], confidence, probabilities
