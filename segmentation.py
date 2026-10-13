"""Pixel-level dust segmentation for SolarSense.

Training data must contain paired RGB images and binary dust masks. A mask pixel
of 255 means dust; 0 means non-dust. This module never treats Grad-CAM as a
segmentation ground truth.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import transforms

try:
    from .gradcam import detect_panel_mask
    from .model import DEVICE
except ImportError:  # pragma: no cover - script execution fallback
    from gradcam import detect_panel_mask
    from model import DEVICE

SEGMENTATION_SIZE = 256


class DoubleConv(nn.Module):
    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = DoubleConv(3, 32)
        self.down2 = DoubleConv(32, 64)
        self.down3 = DoubleConv(64, 128)
        self.down4 = DoubleConv(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(256, 512)
        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv4 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv3 = DoubleConv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv2 = DoubleConv(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv1 = DoubleConv(64, 32)
        self.output = nn.Conv2d(32, 1, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        first = self.down1(inputs)
        second = self.down2(self.pool(first))
        third = self.down3(self.pool(second))
        fourth = self.down4(self.pool(third))
        center = self.bottleneck(self.pool(fourth))
        x = self.up4(center)
        x = self.conv4(torch.cat([x, fourth], dim=1))
        x = self.up3(x)
        x = self.conv3(torch.cat([x, third], dim=1))
        x = self.up2(x)
        x = self.conv2(torch.cat([x, second], dim=1))
        x = self.up1(x)
        x = self.conv1(torch.cat([x, first], dim=1))
        return self.output(x)


def load_segmentation_model(path: str | Path) -> UNet:
    model = UNet().to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    model.eval()
    return model


def predict_dust_mask(
    model: UNet,
    image_bgr: np.ndarray,
    threshold: float = 0.5,
) -> tuple[np.ndarray, float]:
    """Return a full-size binary dust mask and panel-relative coverage percent."""
    height, width = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    tensor = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((SEGMENTATION_SIZE, SEGMENTATION_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])(image_rgb).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probabilities = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()
    mask = cv2.resize((probabilities >= threshold).astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
    panel = detect_panel_mask(image_bgr) > 0
    coverage = float((mask.astype(bool) & panel).sum()) / max(int(panel.sum()), 1) * 100.0
    return mask * 255, round(coverage, 1)


def dice_score(prediction: torch.Tensor, target: torch.Tensor) -> float:
    prediction = prediction.bool()
    target = target.bool()
    intersection = (prediction & target).sum().item()
    return (2.0 * intersection + 1.0) / (prediction.sum().item() + target.sum().item() + 1.0)


def iou_score(prediction: torch.Tensor, target: torch.Tensor) -> float:
    prediction = prediction.bool()
    target = target.bool()
    intersection = (prediction & target).sum().item()
    union = (prediction | target).sum().item()
    return (intersection + 1.0) / (union + 1.0)
