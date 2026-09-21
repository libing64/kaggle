#!/usr/bin/env python3
"""Disaster Tweets – naive TF-IDF + Logistic Regression."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "data" / "nlp-getting-started"


def main() -> None:
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = train["target"].astype(int)
    # keyword is a cheap extra token; location is too noisy for a naive pass.
    text = (train["keyword"].fillna("") + " " + train["text"].fillna("")).str.strip()
    text_test = (test["keyword"].fillna("") + " " + test["text"].fillna("")).str.strip()

    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=20000,
                ),
            ),
            ("lr", LogisticRegression(C=1.0, max_iter=200, solver="liblinear")),
        ]
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1 = cross_val_score(model, text, y, cv=cv, scoring="f1", n_jobs=-1)
    print(f"5-fold F1: mean={f1.mean():.4f}  std={f1.std():.4f}  folds={np.round(f1, 4)}")

    model.fit(text, y)
    pred = model.predict(text_test)
    out = pd.DataFrame({"id": test["id"], "target": pred.astype(int)})
    path = ROOT / "submission.csv"
    out.to_csv(path, index=False)
    print(f"wrote {path}  rows={len(out)}")


if __name__ == "__main__":
    main()
