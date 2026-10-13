"""Dust-class Grad-CAM for the supplied PyTorch MobileNetV2 model.

Important fixes over the old implementation:
* Uses PyTorch hooks, because the supplied checkpoint is PyTorch.
* Explains the Dusty class logit instead of treating a two-class model as a
  sigmoid model.  This makes the result a dust map even when the prediction
  is Clean.
* Uses ImageNet normalization, matching the classifier's preprocessing.
* Applies the panel mask only to the heatmap.  The old blend darkened the
  entire non-panel background because it blended black heatmap pixels into it.
"""

from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np
import torch

try:
    from .model import preprocess_rgb, predict_tensor
except ImportError:  # pragma: no cover - script execution fallback
    from model import preprocess_rgb, predict_tensor


# Full-resolution 4K frames create several temporary float arrays during
# Grad-CAM.  Cap the rendered analysis size while keeping the original upload
# available through the API.  This prevents avoidable out-of-memory failures.
MAX_RENDER_SIDE = 1920


def read_bgr(image_or_path: str | Path | np.ndarray) -> np.ndarray:
    """Read an image as uint8 BGR, accepting a path or an in-memory frame."""
    if isinstance(image_or_path, (str, Path)):
        image = cv2.imread(str(image_or_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_or_path}")
    else:
        image = np.asarray(image_or_path)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Expected a BGR image with shape (H, W, 3).")
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
    height, width = image.shape[:2]
    largest_side = max(height, width)
    if largest_side > MAX_RENDER_SIDE:
        scale = MAX_RENDER_SIDE / float(largest_side)
        image = cv2.resize(
            image,
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    return image.copy()


def detect_panel_mask(image_bgr: np.ndarray) -> np.ndarray:
    """Return a binary mask for the dominant panel-shaped region.

    A contour is preferred when a large quadrilateral is visible.  The
    central 90% fallback is deliberately conservative: it prevents an
    unstable edge detector from making the coverage denominator zero.
    """
    height, width = image_bgr.shape[:2]
    image_area = float(height * width)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 140)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(
        closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    best_contour = None
    best_area = 0.0
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < 0.10 * image_area:
            continue
        perimeter = cv2.arcLength(contour, True)
        approximation = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
        if 4 <= len(approximation) <= 8 and area > best_area:
            best_contour = contour
            best_area = area

    mask = np.zeros((height, width), dtype=np.uint8)
    if best_contour is not None:
        cv2.drawContours(mask, [best_contour], -1, 255, thickness=-1)
    else:
        margin_y = max(1, int(height * 0.05))
        margin_x = max(1, int(width * 0.05))
        mask[margin_y : height - margin_y, margin_x : width - margin_x] = 255

    # Small holes in a panel contour should not become artificial clean areas.
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
    )
    return mask


class GradCAM:
    """Hook-based Grad-CAM implementation for a PyTorch classification model."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._handles = [
            target_layer.register_forward_hook(self._save_activations),
            target_layer.register_full_backward_hook(self._save_gradients),
        ]

    def _save_activations(
        self, _module: torch.nn.Module, _inputs: tuple, output: torch.Tensor
    ) -> None:
        self.activations = output

    def _save_gradients(
        self, _module: torch.nn.Module, _inputs: tuple, output: tuple[torch.Tensor]
    ) -> None:
        self.gradients = output[0]

    def remove_hooks(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __call__(
        self, image_tensor: torch.Tensor, target_class: int | None = None
    ) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        self.activations = None
        self.gradients = None

        with torch.enable_grad():
            logits = self.model(image_tensor)
            if target_class is None:
                target_class = int(torch.argmax(logits, dim=1).item())
            score = logits[:, target_class].sum()
            score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not receive activations/gradients.")

        activations = self.activations.detach()[0]
        gradients = self.gradients.detach()[0]
        weights = gradients.mean(dim=(1, 2), keepdim=True)
        cam = (weights * activations).sum(dim=0).clamp(min=0)
        cam -= cam.min()
        maximum = float(cam.max().item())
        if maximum <= 1e-8:
            return np.zeros((1, 1), dtype=np.float32)
        return (cam / maximum).cpu().numpy().astype(np.float32)


def _find_target_layer(model: torch.nn.Module) -> torch.nn.Module:
    """Use the final convolutional feature map of MobileNetV2."""
    try:
        return model.features[-1]
    except (AttributeError, IndexError) as exc:
        raise ValueError("The model does not expose MobileNetV2 features[-1].") from exc


def apply_dust_heatmap(
    image_or_path: str | Path | np.ndarray,
    model: torch.nn.Module,
    threshold: float = 0.35,
    alpha: float = 0.70,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Create an RGB overlay, RGB heatmap, and panel-only coverage percentage.

    The first two outputs use the rendered analysis dimensions (capped at
    1920px on the largest side). The heatmap is black outside the panel ROI;
    the overlay keeps the original background unchanged and blends color only
    where the panel heatmap has activation.
    """
    image_bgr = read_bgr(image_or_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_tensor = preprocess_rgb(image_rgb)

    predicted_index, _confidence, _probabilities = predict_tensor(
        model, image_tensor
    )
    # Explain the class the model actually predicted instead of forcing a Dusty
    # explanation for every image. This keeps the heatmap aligned with the
    # classification result and prevents clean panels from showing dust-like
    # activation maps.
    cam_runner = GradCAM(model, _find_target_layer(model))
    try:
        raw_cam = cam_runner(image_tensor, target_class=int(predicted_index))
    finally:
        cam_runner.remove_hooks()

    height, width = image_bgr.shape[:2]
    cam = cv2.resize(raw_cam, (width, height), interpolation=cv2.INTER_CUBIC)
    cam = cv2.GaussianBlur(cam, (0, 0), sigmaX=max(width, height) / 300.0)
    cam = np.clip(cam, 0.0, 1.0)

    panel_mask = detect_panel_mask(image_bgr)
    panel = panel_mask > 0
    masked_cam = np.where(panel, cam, 0.0).astype(np.float32)
    panel_values = masked_cam[panel]
    if panel_values.size and float(panel_values.max()) > 1e-8:
        # Re-normalize only within the panel, so outside pixels cannot affect
        # the color scale or the dust coverage percentage.
        masked_cam = masked_cam / float(panel_values.max())

    dust_binary = panel & (masked_cam >= float(threshold))
    panel_pixel_count = int(panel.sum())
    dust_coverage = (
        float(dust_binary.sum()) / max(panel_pixel_count, 1) * 100.0
    )

    heatmap_bgr = cv2.applyColorMap(
        np.uint8(np.clip(masked_cam, 0.0, 1.0) * 255.0),
        cv2.COLORMAP_INFERNO,
    )
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    heatmap_rgb[~panel] = 0

    # Preserve the original background.  Opacity grows with activation so
    # low-response panel pixels remain visible instead of becoming blue noise.
    activation_opacity = (
        np.clip(masked_cam * float(alpha), 0.0, float(alpha)) * panel
    )[..., None]
    overlay = (
        image_rgb.astype(np.float32) * (1.0 - activation_opacity)
        + heatmap_rgb.astype(np.float32) * activation_opacity
    ).clip(0, 255).astype(np.uint8)

    return overlay, heatmap_rgb, dust_coverage
