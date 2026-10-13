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
import json
from pathlib import Path

import matplotlib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

matplotlib.use("Agg")
from matplotlib import pyplot as plt

try:
    from .model import CLASS_NAMES, DEVICE, IMAGE_SIZE, build_model
except ImportError:  # pragma: no cover - script execution fallback
    from model import CLASS_NAMES, DEVICE, IMAGE_SIZE, build_model


class _BinaryImageFolder(datasets.ImageFolder):
    """Read only the classifier's classes while ignoring unrelated folders."""

    def find_classes(self, directory: str) -> tuple[list[str], dict[str, int]]:
        available = {path.name for path in Path(directory).iterdir() if path.is_dir()}
        missing = [name for name in CLASS_NAMES if name not in available]
        if missing:
            raise ValueError(
                f"Missing class folders {tuple(missing)} under {directory}."
            )
        classes = list(CLASS_NAMES)
        return classes, {name: index for index, name in enumerate(classes)}


def split_flat_dataset(
    dataset_dir: Path,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[Subset, Subset, tuple[str, ...]]:
    """Create reproducible stratified train/test subsets from Clean/Dusty folders."""
    _, eval_transform = make_transforms()
    train_transform, _ = make_transforms()
    base_train = _BinaryImageFolder(str(dataset_dir), transform=train_transform)
    base_eval = _BinaryImageFolder(str(dataset_dir), transform=eval_transform)
    expected = tuple(CLASS_NAMES)

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
    if len(y_true) != len(y_pred):
        raise ValueError("Prediction and target counts must match.")

    matrix = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred):
        matrix[true_label, pred_label] += 1
    total = matrix.sum()
    if total != sample_count:
        raise ValueError(
            f"Expected {sample_count} prediction pairs but got {total}."
        )
    per_class = {}
    precision_values = []
    recall_values = []
    f1_values = []
    for index in range(len(CLASS_NAMES)):
        tp = int(matrix[index, index])
        fp = int(matrix[:, index].sum() - tp)
        fn = int(matrix[index, :].sum() - tp)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)
        per_class[CLASS_NAMES[index]] = {
            "precision": round(precision * 100.0, 2),
            "recall": round(recall * 100.0, 2),
            "f1_score": round(f1 * 100.0, 2),
            "support": int(matrix[index, :].sum()),
        }

    macro_precision = float(np.mean(precision_values))
    macro_recall = float(np.mean(recall_values))
    macro_f1 = float(np.mean(f1_values))
    positive_index = CLASS_NAMES.index("Dusty")
    negative_index = CLASS_NAMES.index("Clean")
    true_positive = int(matrix[positive_index, positive_index])
    false_negative = int(matrix[positive_index, negative_index])
    false_positive = int(matrix[negative_index, positive_index])
    true_negative = int(matrix[negative_index, negative_index])
    binary_precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    binary_recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    binary_f1 = (
        2 * binary_precision * binary_recall / (binary_precision + binary_recall)
        if binary_precision + binary_recall
        else 0.0
    )
    specificity = (
        true_negative / (true_negative + false_positive)
        if true_negative + false_positive
        else 0.0
    )
    return {
        "accuracy": round(float(matrix.diagonal().sum() / total) * 100.0, 2) if total else 0.0,
        "f1_score": round(binary_f1 * 100.0, 2),
        "precision": round(binary_precision * 100.0, 2),
        "recall": round(binary_recall * 100.0, 2),
        "specificity": round(specificity * 100.0, 2),
        "positive_class": CLASS_NAMES[positive_index],
        "metric_average": "binary",
        "macro_precision": round(macro_precision * 100.0, 2),
        "macro_recall": round(macro_recall * 100.0, 2),
        "macro_f1_score": round(macro_f1 * 100.0, 2),
        "true_positive": true_positive,
        "false_negative": false_negative,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "confusion_matrix_orientation": "rows=actual, columns=predicted",
        "confusion_matrix": matrix.tolist(),
        "class_names": list(CLASS_NAMES),
        "sample_count": int(sample_count),
        "per_class": per_class,
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

    return _metrics_from_predictions(
        y_true, y_pred, dataset_dir, len(dataset)
    )


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
    metrics = _metrics_from_predictions(
        y_true, y_pred, dataset_dir, len(test_set)
    )
    metrics["test_fraction"] = test_fraction
    metrics["validation_method"] = "deterministic stratified held-out split (seed=42)"
    return metrics


def evaluate_test_set(
    model: nn.Module,
    loader: DataLoader,
    dataset_dir: Path,
    criterion: nn.Module,
) -> tuple[float, dict[str, object]]:
    """Evaluate a held-out loader and derive class metrics from its predictions."""
    model.eval()
    total_loss = 0.0
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            logits = model(images)
            total_loss += float(criterion(logits, labels).item()) * labels.size(0)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().tolist())

    sample_count = len(loader.dataset)
    metrics = _metrics_from_predictions(y_true, y_pred, dataset_dir, sample_count)
    return total_loss / max(sample_count, 1), metrics


def save_training_plots(
    history: list[dict[str, float | int | str]],
    metrics: dict[str, object],
    output: Path,
) -> tuple[Path, Path]:
    """Save loss curves and a labeled held-out confusion matrix."""
    output.parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history) + 1)

    loss_path = output.with_name(f"{output.stem}_loss.png")
    plt.figure(figsize=(9, 5))
    plt.plot(epochs, [item["train_loss"] for item in history], label="Training loss")
    plt.plot(epochs, [item["validation_loss"] for item in history], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title("MobileNetV2 training vs validation loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(loss_path, dpi=160)
    plt.close()

    matrix_path = output.with_name(f"{output.stem}_confusion_matrix.png")
    confusion_matrix = np.asarray(metrics["confusion_matrix"])
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(confusion_matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    names = metrics["class_names"]
    axis.set(
        xticks=range(len(names)),
        yticks=range(len(names)),
        xticklabels=names,
        yticklabels=names,
        xlabel="Predicted panel condition",
        ylabel="Actual panel condition",
        title="Held-out test confusion matrix",
    )
    threshold = confusion_matrix.max() / 2 if confusion_matrix.size else 0
    for row in range(confusion_matrix.shape[0]):
        for column in range(confusion_matrix.shape[1]):
            axis.text(
                column,
                row,
                str(confusion_matrix[row, column]),
                ha="center",
                va="center",
                color="white" if confusion_matrix[row, column] > threshold else "black",
            )
    figure.tight_layout()
    figure.savefig(matrix_path, dpi=160)
    plt.close(figure)
    return loss_path, matrix_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--train-dir", type=Path, default=Path("dataset/train"))
    parser.add_argument("--test-dir", type=Path, default=Path("dataset/test"))
    parser.add_argument("--output", type=Path, default=Path("model.pth"))
    parser.add_argument("--head-epochs", type=int, default=15)
    parser.add_argument("--fine-tune-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=min(4, torch.get_num_threads()),
        help="Limit CPU threads to avoid oversubscribing small training workloads.",
    )
    args = parser.parse_args()
    if args.cpu_threads < 1:
        parser.error("--cpu-threads must be greater than zero.")
    torch.set_num_threads(args.cpu_threads)

    if args.data_dir.exists() and (args.data_dir / "Clean").exists():
        train_source, test_set, _classes = split_flat_dataset(args.data_dir)
        _, eval_transform = make_transforms()
        eval_source = _BinaryImageFolder(
            str(args.data_dir), transform=eval_transform
        )
        generator = torch.Generator().manual_seed(42)
        indices = torch.randperm(len(train_source), generator=generator).tolist()
        validation_count = max(1, int(len(indices) * 0.15))
        val_indices = indices[:validation_count]
        train_indices = indices[validation_count:]
        train_set = Subset(train_source, train_indices)
        val_set = Subset(eval_source, [train_source.indices[index] for index in val_indices])
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

    model = build_model(pretrained=True).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    best_val_accuracy = -1.0
    history: list[dict[str, float | int | str]] = []
    args.output.parent.mkdir(parents=True, exist_ok=True)

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
        history.append({
            "phase": "head",
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "validation_loss": val_loss,
            "train_accuracy": train_accuracy,
            "validation_accuracy": val_accuracy,
        })
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
        history.append({
            "phase": "fine-tune",
            "epoch": args.head_epochs + epoch + 1,
            "train_loss": train_loss,
            "validation_loss": val_loss,
            "train_accuracy": train_accuracy,
            "validation_accuracy": val_accuracy,
        })
        print(
            f"fine {epoch + 1:02d}/{args.fine_tune_epochs} "
            f"train loss={train_loss:.4f} acc={train_accuracy:.3f} "
            f"val loss={val_loss:.4f} acc={val_accuracy:.3f}"
        )
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), args.output)

    model.load_state_dict(torch.load(args.output, map_location=DEVICE))
    test_loss, metrics = evaluate_test_set(
        model, test_loader, args.data_dir, criterion
    )
    metrics["test_loss"] = round(test_loss, 6)
    metrics["evaluation_type"] = "held-out test set"
    metrics["panel_type"] = "binary panel condition: Clean vs Dusty"
    metrics["training_history"] = history
    metrics_path = args.output.with_name(f"{args.output.stem}_metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    loss_path, matrix_path = save_training_plots(history, metrics, args.output)
    print(f"held-out test loss={test_loss:.4f}")
    print(json.dumps(metrics, indent=2))
    print(f"saved checkpoint: {args.output}")
    print(f"saved metrics: {metrics_path}")
    print(f"saved loss graph: {loss_path}")
    print(f"saved confusion matrix graph: {matrix_path}")


if __name__ == "__main__":
    main()
