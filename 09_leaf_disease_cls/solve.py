#!/usr/bin/env python3
"""Cassava Leaf Disease Classification — EfficientNet-B0 v3.

Improvements vs the first 320px / 8-epoch baseline (~0.788):
- train from ImageNet at 384 with moderate Albumentations
- light Mixup only in the first half of training
- cosine LR, label smoothing, class-balanced sampling
- validation TTA; never overwrite baseline.pt
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "data" / "cassava"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 384
EPOCHS = 16
BATCH = 16
NUM_CLASSES = 5
MIXUP_ALPHA = 0.15
MIXUP_UNTIL = EPOCHS // 2
BASELINE_PATH = ROOT / "efficientnet_b0_baseline.pt"
BEST_PATH = ROOT / "efficientnet_b0_v3.pt"
DEPLOY_PATH = ROOT / "efficientnet_b0.pt"
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class LeafDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, image_dir: Path, transform):
        self.frame = frame.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        image = cv2.imread(str(self.image_dir / row["image_id"]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image=image)["image"]
        if "label" in self.frame.columns:
            return image, int(row["label"])
        return image, row["image_id"]


def transforms_train():
    return A.Compose(
        [
            A.RandomResizedCrop(size=(IMG_SIZE, IMG_SIZE), scale=(0.7, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Affine(
                scale=(0.9, 1.1),
                translate_percent=(-0.05, 0.05),
                rotate=(-25, 25),
                shear=(-5, 5),
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.6,
            ),
            A.OneOf(
                [
                    A.HueSaturationValue(15, 25, 15, p=1.0),
                    A.RandomBrightnessContrast(0.15, 0.15, p=1.0),
                ],
                p=0.7,
            ),
            A.CoarseDropout(
                num_holes_range=(1, 4),
                hole_height_range=(8, 32),
                hole_width_range=(8, 32),
                fill=0,
                p=0.35,
            ),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ]
    )


def transforms_eval():
    return A.Compose(
        [
            A.Resize(IMG_SIZE + 32, IMG_SIZE + 32),
            A.CenterCrop(IMG_SIZE, IMG_SIZE),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ]
    )


def build_model(resume: Path | None = None) -> nn.Module:
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    if resume is not None and resume.exists():
        state = torch.load(resume, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        print(f"resumed from {resume}")
    return model.to(DEVICE)


def class_weights(labels: np.ndarray) -> torch.Tensor:
    counts = np.bincount(labels, minlength=NUM_CLASSES).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    return torch.tensor(weights / weights.mean(), dtype=torch.float32, device=DEVICE)


def sample_weights(labels: np.ndarray) -> list[float]:
    counts = np.bincount(labels, minlength=NUM_CLASSES).astype(np.float64)
    inv = 1.0 / np.maximum(counts, 1.0)
    return (inv[labels] / inv[labels].sum() * len(labels)).tolist()


def mixup_batch(images: torch.Tensor, labels: torch.Tensor, alpha: float):
    if alpha <= 0:
        return images, labels, labels, 1.0
    lam = float(np.random.beta(alpha, alpha))
    index = torch.randperm(images.size(0), device=images.device)
    mixed = lam * images + (1.0 - lam) * images[index]
    return mixed, labels, labels[index], lam


def run_epoch(model, loader, opt=None, criterion=None, scaler=None, use_mixup: bool = False) -> tuple[float, float]:
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
            if use_mixup:
                images, y_a, y_b, lam = mixup_batch(images, labels, MIXUP_ALPHA)
            else:
                y_a = y_b = labels
                lam = 1.0
        with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
            logits = model(images)
            if train and use_mixup:
                loss = lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)
            else:
                loss = criterion(logits, labels)
        if train:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        total_loss += float(loss.item()) * len(labels)
        # Accuracy only meaningful without Mixup; still logged for monitoring.
        correct += int((logits.argmax(1) == labels).sum().item())
        total += len(labels)
    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def predict_proba_tta(model, frame: pd.DataFrame, image_dir: Path) -> tuple[list[str], np.ndarray]:
    model.eval()
    variants = [
        transforms_eval(),
        A.Compose(
            [
                A.Resize(IMG_SIZE + 32, IMG_SIZE + 32),
                A.CenterCrop(IMG_SIZE, IMG_SIZE),
                A.HorizontalFlip(p=1.0),
                A.Normalize(mean=MEAN, std=STD),
                ToTensorV2(),
            ]
        ),
        A.Compose(
            [
                A.Resize(IMG_SIZE + 32, IMG_SIZE + 32),
                A.CenterCrop(IMG_SIZE, IMG_SIZE),
                A.VerticalFlip(p=1.0),
                A.Normalize(mean=MEAN, std=STD),
                ToTensorV2(),
            ]
        ),
        A.Compose(
            [
                A.Resize(IMG_SIZE + 32, IMG_SIZE + 32),
                A.CenterCrop(IMG_SIZE, IMG_SIZE),
                A.HorizontalFlip(p=1.0),
                A.VerticalFlip(p=1.0),
                A.Normalize(mean=MEAN, std=STD),
                ToTensorV2(),
            ]
        ),
    ]
    image_ids = frame["image_id"].tolist()
    probs = np.zeros((len(image_ids), NUM_CLASSES), dtype=np.float64)
    for transform in variants:
        loader = DataLoader(
            LeafDataset(frame, image_dir, transform),
            batch_size=BATCH,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )
        offset = 0
        for images, batch_ids in loader:
            logits = model(images.to(DEVICE, non_blocking=True))
            batch_prob = torch.softmax(logits.float(), dim=1).cpu().numpy()
            probs[offset : offset + len(batch_ids)] += batch_prob
            offset += len(batch_ids)
    probs /= len(variants)
    return image_ids, probs


def main() -> None:
    if not (DATA / "train.csv").exists():
        raise SystemExit(f"missing {DATA / 'train.csv'}; run download.py first")

    if DEPLOY_PATH.exists() and not BASELINE_PATH.exists():
        shutil.copy2(DEPLOY_PATH, BASELINE_PATH)
        print(f"backed up deploy weights -> {BASELINE_PATH}")

    print(f"device={DEVICE}  img={IMG_SIZE}  epochs={EPOCHS}  batch={BATCH}")
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

    train_loader = DataLoader(
        LeafDataset(train_df, DATA / "train_images", transforms_train()),
        batch_size=BATCH,
        sampler=WeightedRandomSampler(sample_weights(train_df["label"].to_numpy()), len(train_df), replacement=True),
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        LeafDataset(val_df, DATA / "train_images", transforms_eval()),
        batch_size=BATCH,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # Fresh ImageNet start — previous fine-tune-from-baseline + heavy Mixup hurt val.
    model = build_model(resume=None)
    criterion = nn.CrossEntropyLoss(weight=class_weights(train_df["label"].to_numpy()), label_smoothing=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")

    best_acc = 0.0
    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        use_mixup = epoch <= MIXUP_UNTIL
        tr_loss, tr_acc = run_epoch(model, train_loader, opt, criterion, scaler, use_mixup=use_mixup)
        torch.cuda.empty_cache()
        va_loss, va_acc = run_epoch(model, val_loader, None, criterion, None, use_mixup=False)
        sched.step()
        print(
            f"epoch {epoch:02d}/{EPOCHS}  mixup={int(use_mixup)}  "
            f"train_loss={tr_loss:.4f} acc={tr_acc:.4f}  "
            f"val_loss={va_loss:.4f} acc={va_acc:.4f}",
            flush=True,
        )
        if va_acc > best_acc:
            best_acc = va_acc
            torch.save(model.state_dict(), BEST_PATH)
            print(f"  saved {BEST_PATH}  best_val_acc={best_acc:.4f}", flush=True)

    print(f"best_val_acc={best_acc:.4f}  train_seconds={time.time() - t0:.1f}")

    model.load_state_dict(torch.load(BEST_PATH, map_location=DEVICE, weights_only=True))
    _, val_probs = predict_proba_tta(model, val_df, DATA / "train_images")
    tta_acc = float((val_probs.argmax(1) == val_df["label"].to_numpy()).mean())
    print(f"val_acc_with_TTA={tta_acc:.4f}")

    if tta_acc >= 0.788 or best_acc >= 0.788:
        shutil.copy2(BEST_PATH, DEPLOY_PATH)
        print(f"updated deploy weights -> {DEPLOY_PATH}")
    else:
        print(f"kept previous deploy weights ({DEPLOY_PATH}); v3 did not beat 0.788")

    test_dir = DATA / "test_images"
    test_ids = sorted(p.name for p in test_dir.glob("*.jpg"))
    if not test_ids:
        test_ids = pd.read_csv(DATA / "sample_submission.csv")["image_id"].tolist()
    test_df = pd.DataFrame({"image_id": test_ids})
    ids, probs = predict_proba_tta(model, test_df, test_dir)
    out = pd.DataFrame({"image_id": ids, "label": probs.argmax(1).astype(int)})
    path = ROOT / "submission.csv"
    out.to_csv(path, index=False)
    print(f"wrote {path} rows={len(out)}")
    print(out["label"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
