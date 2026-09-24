"""Train the same MobileNetV2 architecture used by the inference package.

Expected dataset layout:
    dataset/
      train/clean/*
      train/dusty/*
      test/clean/*
      test/dusty/*

The old training script produced a TensorFlow `.h5` model, which could not
load the supplied PyTorch `.pth` checkpoint.  This script keeps training and
inference on the same architecture and normalization.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from model import CLASS_NAMES, DEVICE, IMAGE_SIZE, build_model


def split_flat_dataset(
    dataset_dir: Path,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[Subset, Subset, tuple[str, ...]]:
    """Create reproducible stratified train/test subsets from Clean/Dusty folders."""
    _, eval_transform = make_transforms()
    train_transform, _ = make_transforms()
    base_train = datasets.ImageFolder(str(dataset_dir), transform=train_transform)
    base_eval = datasets.ImageFolder(str(dataset_dir), transform=eval_transform)
    expected = tuple(CLASS_NAMES)
    if tuple(base_train.classes) != expected:
        raise ValueError(f"Expected class folders {expected}, found {tuple(base_train.classes)}.")

    generator = torch.Generator().manual_seed(seed)
    train_indices: list[int] = []
    test_indices: list[int] = []
    for class_index in range(len(expected)):
        class_indices = [index for index, target in enumerate(base_train.targets) if target == class_index]
        shuffled = torch.randperm(len(class_indices), generator=generator).tolist()
        test_count = max(1, int(round(len(class_indices) * test_fraction)))
        test_indices.extend(class_indices[position] for position in shuffled[:test_count])
        train_indices.extend(class_indices[position] for position in shuffled[test_count:])
    return Subset(base_train, train_indices), Subset(base_eval, test_indices), expected


def _metrics_from_predictions(
    y_true: list[int],
    y_pred: list[int],
    dataset_dir: Path,
    sample_count: int,
) -> dict[str, object]:
    matrix = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred):
        matrix[true_label, pred_label] += 1
    total = matrix.sum()
    precision_values = []
    recall_values = []
    f1_values = []
    for index in range(len(CLASS_NAMES)):
        tp = int(matrix[index, index])
        fp = int(matrix[:, index].sum() - tp)
        fn = int(matrix[index, :].sum() - tp)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return {
        "accuracy": round(float(matrix.diagonal().sum() / total) * 100.0, 2) if total else 0.0,
        "f1_score": round(float(np.mean(f1_values)) * 100.0, 2),
        "precision": round(float(np.mean(precision_values)) * 100.0, 2),
        "recall": round(float(np.mean(recall_values)) * 100.0, 2),
        "confusion_matrix": matrix.tolist(),
        "class_names": list(CLASS_NAMES),
        "sample_count": int(sample_count),
        "dataset": str(dataset_dir),
    }


def make_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15),
        transforms.ToTensor(),
        normalize,
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        normalize,
    ])
    return train_transform, eval_transform


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        if training:
            optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        if training:
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item()) * labels.size(0)
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        total += labels.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


def evaluate_directory(
    model: nn.Module,
    dataset_dir: Path,
    batch_size: int = 32,
) -> dict[str, object]:
    """Evaluate a directory of clean/dusty images and return accuracy/F1/confusion metrics."""
    _, eval_transform = make_transforms()
    dataset = datasets.ImageFolder(str(dataset_dir), transform=eval_transform)
    expected_classes = tuple(CLASS_NAMES)
    if tuple(dataset.classes) != expected_classes:
        raise ValueError(
            f"Expected class folders {expected_classes}, found {tuple(dataset.classes)}."
        )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            logits = model(images)
            preds = logits.argmax(dim=1).cpu()
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.tolist())

    matrix = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred):
        matrix[true_label, pred_label] += 1

    total = matrix.sum()
    accuracy = float(matrix.diagonal().sum() / total) if total else 0.0
    precision_values = []
    recall_values = []
    f1_values = []
    for idx in range(len(CLASS_NAMES)):
        tp = int(matrix[idx, idx])
        fp = int(matrix[:, idx].sum() - tp)
        fn = int(matrix[idx, :].sum() - tp)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )

    precision = float(np.mean(precision_values)) if precision_values else 0.0
    recall = float(np.mean(recall_values)) if recall_values else 0.0
    f1 = float(np.mean(f1_values)) if f1_values else 0.0

    return {
        "accuracy": round(accuracy * 100.0, 2),
        "f1_score": round(f1 * 100.0, 2),
        "precision": round(precision * 100.0, 2),
        "recall": round(recall * 100.0, 2),
        "confusion_matrix": matrix.tolist(),
        "class_names": list(CLASS_NAMES),
        "sample_count": int(total),
        "dataset": str(dataset_dir),
    }


def evaluate_flat_dataset(
    model: nn.Module,
    dataset_dir: Path,
    test_fraction: float = 0.2,
    batch_size: int = 8,
) -> dict[str, object]:
    """Evaluate the deterministic held-out split from dataset/Clean and dataset/Dusty."""
    _train_set, test_set, _classes = split_flat_dataset(dataset_dir, test_fraction)
    loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    y_true: list[int] = []
    y_pred: list[int] = []
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(DEVICE))
            y_true.extend(labels.tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().tolist())
    return _metrics_from_predictions(y_true, y_pred, dataset_dir, len(test_set))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--train-dir", type=Path, default=Path("dataset/train"))
    parser.add_argument("--test-dir", type=Path, default=Path("dataset/test"))
    parser.add_argument("--output", type=Path, default=Path("model.pth"))
    parser.add_argument("--head-epochs", type=int, default=15)
    parser.add_argument("--fine-tune-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    if args.data_dir.exists() and (args.data_dir / "Clean").exists():
        train_source, test_set, _classes = split_flat_dataset(args.data_dir)
        generator = torch.Generator().manual_seed(42)
        indices = torch.randperm(len(train_source), generator=generator).tolist()
        validation_count = max(1, int(len(indices) * 0.15))
        val_indices = indices[:validation_count]
        train_indices = indices[validation_count:]
        train_set = Subset(train_source, train_indices)
        val_set = Subset(train_source, val_indices)
    else:
        train_transform, eval_transform = make_transforms()
        train_source = datasets.ImageFolder(args.train_dir, transform=train_transform)
        eval_source = datasets.ImageFolder(args.train_dir, transform=eval_transform)
        if tuple(train_source.classes) != tuple(CLASS_NAMES):
            raise ValueError(f"Expected class folders {CLASS_NAMES}, found {tuple(train_source.classes)}.")
        generator = torch.Generator().manual_seed(42)
        indices = torch.randperm(len(train_source), generator=generator).tolist()
        validation_count = max(1, int(len(indices) * 0.15))
        val_indices = indices[:validation_count]
        train_indices = indices[validation_count:]
        train_set = Subset(train_source, train_indices)
        val_set = Subset(eval_source, val_indices)
        test_set = datasets.ImageFolder(args.test_dir, transform=eval_transform)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    model = build_model().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    best_val_accuracy = -1.0

    # Train only the replacement classifier first.
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    optimizer = torch.optim.Adam(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=1e-3,
    )
    for epoch in range(args.head_epochs):
        train_loss, train_accuracy = run_epoch(
            model, train_loader, criterion, optimizer
        )
        val_loss, val_accuracy = run_epoch(model, val_loader, criterion, None)
        print(
            f"head {epoch + 1:02d}/{args.head_epochs} "
            f"train loss={train_loss:.4f} acc={train_accuracy:.3f} "
            f"val loss={val_loss:.4f} acc={val_accuracy:.3f}"
        )
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), args.output)

    # Fine-tune the last feature blocks with a smaller learning rate.
    for parameter in model.features.parameters():
        parameter.requires_grad = True
    for block in model.features[:-4].parameters():
        block.requires_grad = False
    optimizer = torch.optim.Adam(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=1e-5,
    )
    for epoch in range(args.fine_tune_epochs):
        train_loss, train_accuracy = run_epoch(
            model, train_loader, criterion, optimizer
        )
        val_loss, val_accuracy = run_epoch(model, val_loader, criterion, None)
        print(
            f"fine {epoch + 1:02d}/{args.fine_tune_epochs} "
            f"train loss={train_loss:.4f} acc={train_accuracy:.3f} "
            f"val loss={val_loss:.4f} acc={val_accuracy:.3f}"
        )
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), args.output)

    model.load_state_dict(torch.load(args.output, map_location=DEVICE))
    test_loss, test_accuracy = run_epoch(model, test_loader, criterion, None)
    print(f"test loss={test_loss:.4f} accuracy={test_accuracy * 100:.2f}%")
    print(f"saved checkpoint: {args.output}")


if __name__ == "__main__":
    main()
