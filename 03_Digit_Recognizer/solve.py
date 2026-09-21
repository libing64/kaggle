#!/usr/bin/env python3
"""Digit Recognizer – naive 1-hidden-layer MLP (sklearn)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "data" / "digit-recognizer"


def main() -> None:
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = train["label"].to_numpy()
    X = train.drop(columns=["label"]).to_numpy(dtype=np.float32) / 255.0
    X_test = test.to_numpy(dtype=np.float32) / 255.0

    clf = MLPClassifier(
        hidden_layer_sizes=(128,),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=256,
        learning_rate_init=1e-3,
        max_iter=20,
        early_stopping=True,
        n_iter_no_change=3,
        random_state=42,
    )

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    acc = cross_val_score(clf, X, y, cv=cv, scoring="accuracy", n_jobs=1)
    print(f"3-fold Accuracy: mean={acc.mean():.4f}  std={acc.std():.4f}  folds={np.round(acc, 4)}")

    clf.fit(X, y)
    pred = clf.predict(X_test)
    out = pd.DataFrame({"ImageId": np.arange(1, len(pred) + 1), "Label": pred})
    path = ROOT / "submission.csv"
    out.to_csv(path, index=False)
    print(f"wrote {path}  rows={len(out)}")


if __name__ == "__main__":
    main()
