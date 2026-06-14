import os
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import models
from torchvision.models import MobileNet_V2_Weights
from collections import Counter
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------
# Reproducibility
# --------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# --------------------------------
# Config
# --------------------------------
DATASET_DIR = "dataset"
BATCH_SIZE  = 32
HEAD_EPOCHS = 20          # FIX: was 15 — more time for head to converge
FINE_EPOCHS = 25          # FIX: was 15 — fine-tuning needs more epochs at low LR
LR_HEAD     = 1e-3
LR_FINE     = 5e-5        # FIX: was 1e-5 — too low, slows learning without benefit
PATIENCE    = 8           # FIX: was 6 — cosine LR needs more time at tail

# --------------------------------
# Convert RGB
# --------------------------------
class ConvertToRGB:
    def __call__(self, img):
        return img.convert("RGB")

# --------------------------------
# Transforms
# --------------------------------
train_transform = transforms.Compose([
    ConvertToRGB(),
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),              # FIX: added — dust appears from any angle
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),  # FIX: added cutout regularization
])

val_transform = transforms.Compose([
    ConvertToRGB(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# --------------------------------
# Dataset split (70/15/15)
# --------------------------------
dataset = torchvision.datasets.ImageFolder(DATASET_DIR)
n = len(dataset)

indices = torch.randperm(n, generator=torch.Generator().manual_seed(SEED)).tolist()
train_n = int(0.7 * n)
val_n   = int(0.15 * n)

train_idx = indices[:train_n]
val_idx   = indices[train_n:train_n + val_n]
test_idx  = indices[train_n + val_n:]

train_ds = Subset(torchvision.datasets.ImageFolder(DATASET_DIR, transform=train_transform), train_idx)
val_ds   = Subset(torchvision.datasets.ImageFolder(DATASET_DIR, transform=val_transform),   val_idx)
test_ds  = Subset(torchvision.datasets.ImageFolder(DATASET_DIR, transform=val_transform),   test_idx)

# --------------------------------
# Class balancing
# --------------------------------
train_labels   = [dataset.targets[i] for i in train_idx]
counts         = Counter(train_labels)
weights        = {c: len(train_labels) / v for c, v in counts.items()}
sample_weights = [weights[l] for l in train_labels]
sampler        = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                          drop_last=True)   # FIX: drop_last avoids single-sample batch issues
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

print("Class distribution:", dict(counts))
print(f"Train: {train_n}  Val: {val_n}  Test: {n - train_n - val_n}")

# --------------------------------
# Model — MobileNetV2, wider head
# --------------------------------
model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)

for p in model.features.parameters():
    p.requires_grad = False

# FIX: classifier indices now match what gets saved to model.pth
# index 0=Dropout, 1=Linear(1280→256), 2=ReLU, 3=Dropout, 4=Linear(256→2)
model.classifier = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(model.last_channel, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 2),
)

model = model.to(device)
print("Classifier indices:", {i: type(m).__name__ for i, m in enumerate(model.classifier)})

# FIX: label_smoothing=0.1 — slightly higher than 0.05 for better generalization
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# --------------------------------
# Helpers
# --------------------------------
def run_epoch(loader, optimizer=None):
    if optimizer:
        model.train()
    else:
        model.eval()
    total = 0.0
    ctx = torch.enable_grad() if optimizer else torch.no_grad()
    with ctx:
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if optimizer:
                optimizer.zero_grad()
            out  = model(x)
            loss = criterion(out, y)
            if optimizer:
                loss.backward()
                optimizer.step()
            total += loss.item()
    return total / len(loader)


def evaluate(loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total   += y.size(0)
    return correct / total


def full_metrics(loader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            preds.extend(model(x.to(device)).argmax(1).cpu().numpy())
            labels.extend(y.numpy())
    preds  = np.array(preds)
    labels = np.array(labels)
    return (
        (preds == labels).mean() * 100,
        precision_score(labels, preds, average="macro", zero_division=0) * 100,
        recall_score   (labels, preds, average="macro", zero_division=0) * 100,
        f1_score       (labels, preds, average="macro", zero_division=0) * 100,
        confusion_matrix(labels, preds),
    )


def train_phase(tag, epochs, optimizer, scheduler, patience):
    best_val, best_w, no_imp = float("inf"), None, 0
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        tr  = run_epoch(train_loader, optimizer)
        val = run_epoch(val_loader)
        acc = evaluate(val_loader)

        train_losses.append(tr)
        val_losses.append(val)
        scheduler.step()                        # cosine step per epoch

        flag = ""
        if val < best_val:
            best_val = val
            best_w   = copy.deepcopy(model.state_dict())
            no_imp   = 0
            flag     = "✓"
        else:
            no_imp  += 1

        print(f"  [{tag}] epoch {epoch+1:02d}/{epochs}  "
              f"train={tr:.4f}  val={val:.4f}  acc={acc:.2%}  {flag}")

        if no_imp >= patience:
            print(f"  Early stop at epoch {epoch+1}")
            break

    return best_w, train_losses, val_losses

# --------------------------------
# Phase 1 — head only
# --------------------------------
print("\n=== PHASE 1: HEAD TRAINING ===")
opt1  = torch.optim.AdamW(model.classifier.parameters(), lr=LR_HEAD, weight_decay=1e-4)
# FIX: CosineAnnealingLR instead of no scheduler — smoothly decays LR to near-zero
sch1  = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=HEAD_EPOCHS, eta_min=1e-6)
best_w1, tr1, val1 = train_phase("Head", HEAD_EPOCHS, opt1, sch1, PATIENCE)

# --------------------------------
# Phase 2 — unfreeze last 4 blocks (FIX: was only 2)
# --------------------------------
print("\n=== PHASE 2: FINE-TUNING ===")
model.load_state_dict(best_w1)                  # start from best head weights

for layer in model.features[-4:]:
    for p in layer.parameters():
        p.requires_grad = True

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Trainable params: {trainable:,}")

opt2 = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR_FINE, weight_decay=1e-4,
)
sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=FINE_EPOCHS, eta_min=1e-7)
best_w2, tr2, val2 = train_phase("Finetune", FINE_EPOCHS, opt2, sch2, PATIENCE)

# --------------------------------
# Save
# --------------------------------
model.load_state_dict(best_w2)
torch.save(model.state_dict(), "model.pth")
print("\nSaved → model.pth")

# --------------------------------
# Test results
# --------------------------------
acc, prec, rec, f1, cm = full_metrics(test_loader)

print("\n========================================")
print("  TEST RESULTS")
print("========================================")
print(f"  Accuracy  : {acc:.2f}%")
print(f"  Precision : {prec:.2f}%")
print(f"  Recall    : {rec:.2f}%")
print(f"  F1-Score  : {f1:.2f}%")
print("========================================")

# --------------------------------
# Plots
# --------------------------------
all_tr  = tr1  + tr2
all_val = val1 + val2
p1_len  = len(tr1)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(range(1, len(all_tr)  + 1), all_tr,  marker="o", label="Train loss")
axes[0].plot(range(1, len(all_val) + 1), all_val, marker="o", label="Val loss")
axes[0].axvline(p1_len + 0.5, linestyle="--", color="gray", alpha=0.6, label="Fine-tune start")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_title("Training and validation loss")
axes[0].legend()
axes[0].grid(True)

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=dataset.classes,
            yticklabels=dataset.classes,
            ax=axes[1])
axes[1].set_title(f"Confusion matrix — Test set (Acc={acc:.1f}%)")
axes[1].set_ylabel("Actual")
axes[1].set_xlabel("Predicted")

plt.tight_layout()
plt.savefig("results.png", dpi=150)
print("Plot saved → results.png")