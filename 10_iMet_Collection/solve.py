#!/usr/bin/env python3
"""iMet Collection 2019 — knowledge distillation (teacher → student).

Multi-label attribute tagging (1103 classes), metric = mean F2.

Pipeline:
1) Train a larger teacher (EfficientNet-B3) with BCE on hard labels.
2) Distill into EfficientNet-B0 with
       L = α * BCE(student, y) + (1-α) * T² * BCE(student, sigmoid(teacher/T)).
3) Threshold-search on validation F2; write submission.csv.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 1103
IMG_SIZE = 224
BATCH = 48
TEACHER_NAME = "tf_efficientnet_b3.ns_jft_in1k"
STUDENT_NAME = "tf_efficientnet_b0.ns_jft_in1k"
TEACHER_EPOCHS = 6
STUDENT_EPOCHS = 8
LR_TEACHER = 2e-4
LR_STUDENT = 3e-4
DISTILL_ALPHA = 0.5
DISTILL_T = 3.0
NUM_WORKERS = 4
SEED = 42
THRESHOLDS = np.linspace(0.05, 0.45, 17)

TEACHER_CKPT = ROOT / "teacher_b3.pt"
STUDENT_CKPT = ROOT / "student_b0.pt"
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class IMetDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, image_dir: Path, transform, soft: np.ndarray | None = None):
        self.ids = frame["id"].tolist()
        self.image_dir = image_dir
        self.transform = transform
        self.soft = soft
        self.targets = None
        if "attribute_ids" in frame.columns:
            targets = np.zeros((len(frame), NUM_CLASSES), dtype=np.float32)
            for i, raw in enumerate(frame["attribute_ids"].astype(str)):
                if raw and raw != "nan":
                    for tok in raw.split():
                        targets[i, int(tok)] = 1.0
            self.targets = targets

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int):
        path = self.image_dir / f"{self.ids[index]}.png"
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        if self.targets is None:
            return image, self.ids[index]
        if self.soft is None:
            return image, torch.from_numpy(self.targets[index])
        return image, torch.from_numpy(self.targets[index]), torch.from_numpy(self.soft[index])


def transforms_train():
    return transforms.Compose(
        [
            transforms.Resize(IMG_SIZE + 32),
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )


def transforms_eval():
    return transforms.Compose(
        [
            transforms.Resize(IMG_SIZE + 32),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )


def build_model(name: str) -> nn.Module:
    return timm.create_model(name, pretrained=True, num_classes=NUM_CLASSES).to(DEVICE)


def f2_score(y_true: np.ndarray, y_prob: np.ndarray, thr: float) -> float:
    pred = y_prob >= thr
    # Per-sample F2 then mean (competition metric).
    tp = (pred & (y_true == 1)).sum(axis=1).astype(np.float64)
    fp = (pred & (y_true == 0)).sum(axis=1).astype(np.float64)
    fn = ((~pred) & (y_true == 1)).sum(axis=1).astype(np.float64)
    beta2 = 4.0
    precision = tp / np.maximum(tp + fp, 1e-9)
    recall = tp / np.maximum(tp + fn, 1e-9)
    f2 = (1 + beta2) * precision * recall / np.maximum(beta2 * precision + recall, 1e-9)
    # Samples with no positives still contribute 0 unless both empty → define as 1 if both empty.
    both_empty = (tp + fp + fn) == 0
    f2[both_empty] = 1.0
    return float(f2.mean())


def best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    scores = [(thr, f2_score(y_true, y_prob, thr)) for thr in THRESHOLDS]
    thr, score = max(scores, key=lambda x: x[1])
    return float(thr), float(score)


def run_epoch_supervised(model, loader, opt=None, scaler=None) -> float:
    train = opt is not None
    model.train(train)
    criterion = nn.BCEWithLogitsLoss()
    total = 0.0
    n = 0
    for images, targets in loader:
        images = images.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)
        if train:
            opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, targets)
        if train:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        total += float(loss.item()) * len(images)
        n += len(images)
    return total / max(n, 1)


def run_epoch_distill(model, teacher, loader, opt, scaler) -> float:
    model.train()
    teacher.eval()
    bce = nn.BCEWithLogitsLoss()
    total = 0.0
    n = 0
    for batch in loader:
        if len(batch) == 3:
            images, hard, soft = batch
            soft = soft.to(DEVICE, non_blocking=True)
        else:
            images, hard = batch
            soft = None
        images = images.to(DEVICE, non_blocking=True)
        hard = hard.to(DEVICE, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
            logits = model(images)
            hard_loss = bce(logits, hard)
            if soft is None:
                with torch.no_grad():
                    t_logits = teacher(images)
                    soft = torch.sigmoid(t_logits.float() / DISTILL_T)
            # Temperature-scaled soft BCE.
            soft_loss = bce(logits / DISTILL_T, soft) * (DISTILL_T ** 2)
            loss = DISTILL_ALPHA * hard_loss + (1.0 - DISTILL_ALPHA) * soft_loss
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        total += float(loss.item()) * len(images)
        n += len(images)
    return total / max(n, 1)


@torch.no_grad()
def predict_proba(model, loader) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs = []
    labels = []
    for images, targets in loader:
        images = images.to(DEVICE, non_blocking=True)
        logits = model(images)
        probs.append(torch.sigmoid(logits.float()).cpu().numpy())
        labels.append(targets.numpy())
    return np.concatenate(probs, axis=0), np.concatenate(labels, axis=0)


@torch.no_grad()
def predict_proba_ids(model, frame: pd.DataFrame, image_dir: Path) -> tuple[list[str], np.ndarray]:
    loader = DataLoader(
        IMetDataset(frame, image_dir, transforms_eval()),
        batch_size=BATCH,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    model.eval()
    probs = []
    ids: list[str] = []
    for images, batch_ids in loader:
        images = images.to(DEVICE, non_blocking=True)
        logits = model(images)
        probs.append(torch.sigmoid(logits.float()).cpu().numpy())
        ids.extend(batch_ids)
    return ids, np.concatenate(probs, axis=0)


def train_teacher(train_df: pd.DataFrame, val_df: pd.DataFrame) -> nn.Module:
    print(f"\n=== teacher {TEACHER_NAME} ===", flush=True)
    model = build_model(TEACHER_NAME)
    opt = torch.optim.AdamW(model.parameters(), lr=LR_TEACHER, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=TEACHER_EPOCHS)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")

    train_loader = DataLoader(
        IMetDataset(train_df, DATA / "train", transforms_train()),
        batch_size=max(BATCH // 2, 8),
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        IMetDataset(val_df, DATA / "train", transforms_eval()),
        batch_size=BATCH,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    best_f2 = -1.0
    for epoch in range(1, TEACHER_EPOCHS + 1):
        t0 = time.time()
        tr_loss = run_epoch_supervised(model, train_loader, opt, scaler)
        probs, y_true = predict_proba(model, val_loader)
        thr, f2 = best_threshold(y_true, probs)
        sched.step()
        print(
            f"teacher epoch {epoch}/{TEACHER_EPOCHS}  loss={tr_loss:.4f}  "
            f"val_f2={f2:.4f} thr={thr:.3f}  sec={time.time() - t0:.1f}",
            flush=True,
        )
        if f2 > best_f2:
            best_f2 = f2
            torch.save({"model": model.state_dict(), "thr": thr, "f2": f2, "name": TEACHER_NAME}, TEACHER_CKPT)
            print(f"  saved {TEACHER_CKPT} f2={best_f2:.4f}", flush=True)
    payload = torch.load(TEACHER_CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    return model.eval()


def train_student(train_df: pd.DataFrame, val_df: pd.DataFrame, teacher: nn.Module) -> tuple[nn.Module, float]:
    print(f"\n=== student {STUDENT_NAME} (distill α={DISTILL_ALPHA}, T={DISTILL_T}) ===", flush=True)
    model = build_model(STUDENT_NAME)
    opt = torch.optim.AdamW(model.parameters(), lr=LR_STUDENT, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STUDENT_EPOCHS)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")

    train_loader = DataLoader(
        IMetDataset(train_df, DATA / "train", transforms_train()),
        batch_size=BATCH,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        IMetDataset(val_df, DATA / "train", transforms_eval()),
        batch_size=BATCH,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    best_f2 = -1.0
    best_thr = 0.2
    for epoch in range(1, STUDENT_EPOCHS + 1):
        t0 = time.time()
        tr_loss = run_epoch_distill(model, teacher, train_loader, opt, scaler)
        probs, y_true = predict_proba(model, val_loader)
        thr, f2 = best_threshold(y_true, probs)
        sched.step()
        print(
            f"student epoch {epoch}/{STUDENT_EPOCHS}  loss={tr_loss:.4f}  "
            f"val_f2={f2:.4f} thr={thr:.3f}  sec={time.time() - t0:.1f}",
            flush=True,
        )
        if f2 > best_f2:
            best_f2 = f2
            best_thr = thr
            torch.save(
                {"model": model.state_dict(), "thr": thr, "f2": f2, "name": STUDENT_NAME},
                STUDENT_CKPT,
            )
            print(f"  saved {STUDENT_CKPT} f2={best_f2:.4f}", flush=True)
    payload = torch.load(STUDENT_CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    return model.eval(), float(payload.get("thr", best_thr))


def format_prediction(prob_row: np.ndarray, thr: float) -> str:
    idx = np.where(prob_row >= thr)[0]
    if len(idx) == 0:
        idx = np.array([int(prob_row.argmax())])
    return " ".join(map(str, idx.tolist()))


def main() -> None:
    train_csv = DATA / "train.csv"
    if not train_csv.exists():
        raise SystemExit(f"missing {train_csv}; run download.py first")

    print(f"device={DEVICE}  img={IMG_SIZE}  batch={BATCH}")
    train = pd.read_csv(train_csv)
    labels = pd.read_csv(DATA / "labels.csv") if (DATA / "labels.csv").exists() else None
    print(f"train rows={len(train)}  classes={NUM_CLASSES}")
    if labels is not None:
        print(f"label map rows={len(labels)}")

    # Stratify by clipped tag-count; fall back to plain split if a bin is too rare.
    train["n_attr"] = train["attribute_ids"].astype(str).map(lambda s: min(len(s.split()), 10))
    try:
        train_df, val_df = train_test_split(
            train, test_size=0.1, random_state=SEED, stratify=train["n_attr"]
        )
    except ValueError:
        train_df, val_df = train_test_split(train, test_size=0.1, random_state=SEED)
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    print(f"split train={len(train_df)} val={len(val_df)}")

    if TEACHER_CKPT.exists():
        print(f"loading existing teacher {TEACHER_CKPT}")
        payload = torch.load(TEACHER_CKPT, map_location="cpu", weights_only=False)
        teacher = build_model(payload.get("name", TEACHER_NAME))
        teacher.load_state_dict(payload["model"])
        teacher.eval()
    else:
        teacher = train_teacher(train_df, val_df)

    student, thr = train_student(train_df, val_df, teacher)
    print(f"deploy threshold={thr:.3f}")

    test_dir = DATA / "test"
    sample = pd.read_csv(DATA / "sample_submission.csv")
    if test_dir.exists():
        test_ids = sorted(p.stem for p in test_dir.glob("*.png"))
    else:
        test_ids = sample["id"].tolist()
    test_df = pd.DataFrame({"id": test_ids})
    ids, probs = predict_proba_ids(student, test_df, test_dir)
    pred = [format_prediction(probs[i], thr) for i in range(len(ids))]
    out = pd.DataFrame({"id": ids, "attribute_ids": pred})
    path = ROOT / "submission.csv"
    out.to_csv(path, index=False)
    meta = {
        "teacher": TEACHER_NAME,
        "student": STUDENT_NAME,
        "threshold": thr,
        "distill_alpha": DISTILL_ALPHA,
        "distill_T": DISTILL_T,
        "rows": len(out),
    }
    (ROOT / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {path} rows={len(out)}")
    print(out.head().to_string(index=False))


if __name__ == "__main__":
    main()
