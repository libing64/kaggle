#!/usr/bin/env python3
"""Digit Recognizer – small CNN (conv + batch-norm + light augmentation)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "data" / "digit-recognizer"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SmallCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def shift_batch(x: torch.Tensor) -> torch.Tensor:
    # Translate by at most 2 pixels. Roll is enough for a naive augmentation.
    out = x
    for axis, limit in ((2, 2), (3, 2)):
        delta = int(torch.randint(-limit, limit + 1, ()).item())
        if delta:
            out = torch.roll(out, shifts=delta, dims=axis)
            if delta > 0:
                out.narrow(axis, 0, delta).zero_()
            else:
                out.narrow(axis, delta, -delta).zero_()
    return out


def run_epoch(model: nn.Module, loader: DataLoader, opt: torch.optim.Optimizer | None) -> float:
    train = opt is not None
    model.train(train)
    correct = 0
    total = 0
    loss_fn = nn.CrossEntropyLoss()
    for xb, yb in loader:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)
        if train:
            xb = shift_batch(xb)
            opt.zero_grad()
        logits = model(xb)
        if train:
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
        correct += (logits.argmax(1) == yb).sum().item()
        total += len(yb)
    return correct / total


def fit_predict(x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray, epochs: int) -> np.ndarray:
    model = SmallCNN().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    loader = DataLoader(ds, batch_size=128, shuffle=True)
    for _ in range(epochs):
        run_epoch(model, loader, opt)
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(x_eval), 512):
            xb = torch.from_numpy(x_eval[start : start + 512]).to(DEVICE)
            preds.append(model(xb).argmax(1).cpu().numpy())
    return np.concatenate(preds)


def main() -> None:
    print(f"device: {DEVICE}")
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = train["label"].to_numpy()
    x = train.drop(columns=["label"]).to_numpy(dtype=np.float32).reshape(-1, 1, 28, 28) / 255.0
    x_test = test.to_numpy(dtype=np.float32).reshape(-1, 1, 28, 28) / 255.0

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = []
    for fold, (tr, va) in enumerate(cv.split(x, y), start=1):
        pred = fit_predict(x[tr], y[tr], x[va], epochs=5)
        acc = float((pred == y[va]).mean())
        scores.append(acc)
        print(f"fold {fold}: {acc:.4f}")
    scores_arr = np.array(scores)
    print(f"3-fold Accuracy: mean={scores_arr.mean():.4f}  std={scores_arr.std():.4f}  folds={np.round(scores_arr, 4)}")

    pred = fit_predict(x, y, x_test, epochs=8)
    out = pd.DataFrame({"ImageId": np.arange(1, len(pred) + 1), "Label": pred})
    path = ROOT / "submission.csv"
    out.to_csv(path, index=False)
    print(f"wrote {path}  rows={len(out)}")


if __name__ == "__main__":
    main()
