"""Create binary dust masks for U-Net training.

Controls in the image window:
  Left mouse drag: paint dust
  Right mouse drag: erase dust
  S: save mask and open next image
  N: skip image
  Q or Esc: quit

Masks are saved as grayscale PNG files: 255=dust, 0=non-dust.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


class MaskEditor:
    def __init__(self, image: np.ndarray, mask: np.ndarray, window: str):
        self.image = image
        self.mask = mask
        self.window = window
        self.drawing = False
        self.erase = False
        self.radius = max(4, min(image.shape[:2]) // 100)

    def mouse(self, event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing, self.erase = True, False
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.drawing, self.erase = True, True
        elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
            self.drawing = False
        if self.drawing:
            cv2.circle(self.mask, (x, y), self.radius, 0 if self.erase else 255, -1)

    def render(self) -> np.ndarray:
        display = self.image.copy()
        overlay = np.zeros_like(display)
        overlay[:, :, 2] = self.mask
        return cv2.addWeighted(display, 0.7, overlay, 0.3, 0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', type=Path, default=Path('segmentation/images'))
    parser.add_argument('--masks', type=Path, default=Path('segmentation/masks'))
    args = parser.parse_args()
    args.masks.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(path for path in args.images.iterdir() if path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'})
    if not image_paths:
        raise SystemExit(f'No images found in {args.images}')

    for image_path in image_paths:
        mask_path = args.masks / f'{image_path.stem}.png'
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path.exists() else np.zeros(image.shape[:2], dtype=np.uint8)
        if mask.shape != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        editor = MaskEditor(image, mask, 'SolarSense mask annotation')
        cv2.namedWindow(editor.window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(editor.window, editor.mouse)
        while True:
            cv2.imshow(editor.window, editor.render())
            key = cv2.waitKey(20) & 0xFF
            if key in (ord('s'), ord('S')):
                cv2.imwrite(str(mask_path), editor.mask)
                break
            if key in (ord('n'), ord('N')):
                break
            if key in (ord('q'), ord('Q'), 27):
                cv2.destroyAllWindows()
                return
        cv2.destroyWindow(editor.window)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
