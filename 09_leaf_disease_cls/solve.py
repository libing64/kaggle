#!/usr/bin/env python3
"""Cassava Leaf Disease Classification — EfficientNet-B0 baseline.

Metric: categorization accuracy. Submission columns: image_id,label.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "data" / "cassava"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 320
EPOCHS = 8
BATCH = 16
NUM_CLASSES = 5


class LeafDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, image_dir: Path, transform):
        self.frame = frame.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        image = Image.open(self.image_dir / row["image_id"]).convert("RGB")
        image = self.transform(image)
        if "label" in self.frame.columns:
            return image, int(row["label"])
        return image, row["image_id"]


def transforms_train():
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def transforms_eval():
    return transforms.Compose(
        [
            transforms.Resize(IMG_SIZE + 32),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def build_model() -> nn.Module:
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    return model.to(DEVICE)


def class_weights(labels: np.ndarray) -> torch.Tensor:
    counts = np.bincount(labels, minlength=NUM_CLASSES).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    return torch.tensor(weights / weights.mean(), dtype=torch.float32, device=DEVICE)


def sample_weights(labels: np.ndarray) -> list[float]:
    counts = np.bincount(labels, minlength=NUM_CLASSES).astype(np.float64)
    inv = 1.0 / np.maximum(counts, 1.0)
    return (inv[labels] / inv[labels].sum() * len(labels)).tolist()


def run_epoch(model, loader, opt=None, criterion=None, scaler=None) -> tuple[float, float]:
    train = opt is not None
    model.train(train)
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        if train:
            opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, labels)
        if train:
            if scaler is None:
                loss.backward()
                opt.step()
            else:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
        total_loss += float(loss.item()) * len(labels)
        correct += int((logits.argmax(1) == labels).sum().item())
        total += len(labels)
    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def predict_loader(model, loader) -> tuple[list[str], np.ndarray]:
    model.eval()
    ids, preds = [], []
    for images, image_ids in loader:
        logits = model(images.to(DEVICE, non_blocking=True))
        preds.append(logits.argmax(1).cpu().numpy())
        ids.extend(list(image_ids))
    return ids, np.concatenate(preds)


def main() -> None:
    if not (DATA / "train.csv").exists():
        raise SystemExit(f"missing {DATA / 'train.csv'}; run download.py first")

    print(f"device={DEVICE}")
    train = pd.read_csv(DATA / "train.csv")
    with open(DATA / "label_num_to_disease_map.json") as handle:
        mapping = json.load(handle)
    print("classes", mapping)
    print(train["label"].value_counts().sort_index().to_string())

    labels = train["label"].to_numpy()
    fold = next(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(train, labels))
    tr_idx, va_idx = fold
    train_df = train.iloc[tr_idx].reset_index(drop=True)
    val_df = train.iloc[va_idx].reset_index(drop=True)

    train_ds = LeafDataset(train_df, DATA / "train_images", transforms_train())
    val_ds = LeafDataset(val_df, DATA / "train_images", transforms_eval())
    sampler = WeightedRandomSampler(sample_weights(train_df["label"].to_numpy()), len(train_df), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=BATCH, sampler=sampler, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=2, pin_memory=True)

    model = build_model()
    criterion = nn.CrossEntropyLoss(weight=class_weights(train_df["label"].to_numpy()), label_smoothing=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")

    best_acc = 0.0
    best_path = ROOT / "efficientnet_b0.pt"
    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, opt, criterion, scaler)
        torch.cuda.empty_cache()
        va_loss, va_acc = run_epoch(model, val_loader, None, criterion, None)
        sched.step()
        print(
            f"epoch {epoch:02d}/{EPOCHS}  "
            f"train_loss={tr_loss:.4f} acc={tr_acc:.4f}  "
            f"val_loss={va_loss:.4f} acc={va_acc:.4f}",
            flush=True,
        )
        if va_acc >= best_acc:
            best_acc = va_acc
            torch.save(model.state_dict(), best_path)
    print(f"best_val_acc={best_acc:.4f}  train_seconds={time.time() - t0:.1f}")

    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    test_dir = DATA / "test_images"
    test_ids = sorted(p.name for p in test_dir.glob("*.jpg"))
    if not test_ids:
        # Code competition discloses one sample image locally.
        sample = pd.read_csv(DATA / "sample_submission.csv")
        test_ids = sample["image_id"].tolist()
    test_df = pd.DataFrame({"image_id": test_ids})
    test_loader = DataLoader(
        LeafDataset(test_df, test_dir, transforms_eval()),
        batch_size=BATCH,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    ids, preds = predict_loader(model, test_loader)
    out = pd.DataFrame({"image_id": ids, "label": preds.astype(int)})
    path = ROOT / "submission.csv"
    out.to_csv(path, index=False)
    print(f"wrote {path} rows={len(out)}")
    print(out["label"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
