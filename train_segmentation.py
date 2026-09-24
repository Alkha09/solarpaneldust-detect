"""Train and validate U-Net dust segmentation.

Expected layout:
  segmentation/images/*.jpg
  segmentation/masks/*.png

Each mask must share its image stem and contain 255 for dust pixels and 0
elsewhere. A validation score is meaningful only when these masks are real
pixel annotations, not Grad-CAM exports.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

from model import DEVICE
from segmentation import SEGMENTATION_SIZE, UNet, dice_score, iou_score


class DustMaskDataset(Dataset):
    def __init__(self, images_dir: Path, masks_dir: Path):
        self.images = sorted(path for path in images_dir.iterdir() if path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'})
        self.masks_dir = masks_dir
        self.image_transform = transforms.Compose([
            transforms.Resize((SEGMENTATION_SIZE, SEGMENTATION_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((SEGMENTATION_SIZE, SEGMENTATION_SIZE), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])
        missing = [image.stem for image in self.images if not (masks_dir / f'{image.stem}.png').exists()]
        if missing:
            raise ValueError(f'Missing masks for {len(missing)} images; first missing: {missing[0]}')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        mask_path = self.masks_dir / f'{image_path.stem}.png'
        image = self.image_transform(Image.open(image_path).convert('RGB'))
        mask = (self.mask_transform(Image.open(mask_path).convert('L')) >= 0.5).float()
        return image, mask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', type=Path, default=Path('segmentation/images'))
    parser.add_argument('--masks', type=Path, default=Path('segmentation/masks'))
    parser.add_argument('--output', type=Path, default=Path('segmentation_model.pth'))
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--test-fraction', type=float, default=0.2)
    args = parser.parse_args()

    dataset = DustMaskDataset(args.images, args.masks)
    test_count = max(1, int(round(len(dataset) * args.test_fraction)))
    train_set, test_set = random_split(dataset, [len(dataset) - test_count, test_count], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = UNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_function = nn.BCEWithLogitsLoss()
    best_iou = -1.0

    for epoch in range(args.epochs):
        model.train()
        for images, masks in train_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(images), masks)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            scores = [(dice_score(torch.sigmoid(model(images.to(DEVICE))) >= 0.5, masks.to(DEVICE)), iou_score(torch.sigmoid(model(images.to(DEVICE))) >= 0.5, masks.to(DEVICE))) for images, masks in test_loader]
        dice = sum(score[0] for score in scores) / max(len(scores), 1)
        iou = sum(score[1] for score in scores) / max(len(scores), 1)
        print(f'epoch {epoch + 1:02d}/{args.epochs} dice={dice:.4f} iou={iou:.4f}')
        if iou > best_iou:
            best_iou = iou
            torch.save(model.state_dict(), args.output)
    print(f'saved {args.output}; test dice={dice:.4f}; test iou={iou:.4f}')


if __name__ == '__main__':
    main()
