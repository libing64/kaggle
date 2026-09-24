#!/usr/bin/env python3
"""Disaster Tweets – cleaned text, word + char TF-IDF, logistic regression."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import FeatureUnion, Pipeline

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "data" / "nlp-getting-started"


def clean(text: str) -> str:
    text = str(text).lower()
    text = text.replace("%20", " ")
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = text.replace("&amp;", " ")
    text = re.sub(r"[^a-z0-9#\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def build_text(df: pd.DataFrame) -> pd.Series:
    keyword = df["keyword"].fillna("").map(clean).str.replace(" ", "", regex=False)
    body = df["text"].fillna("").map(clean)
    return ("kw" + keyword + " " + body).str.strip()


def main() -> None:
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = train["target"].astype(int)
    text = build_text(train)
    text_test = build_text(test)

    model = Pipeline(
        [
            (
                "tfidf",
                FeatureUnion(
                    [
                        (
                            "word",
                            TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=30000, sublinear_tf=True),
                        ),
                        (
                            "char",
                            TfidfVectorizer(
                                analyzer="char_wb",
                                ngram_range=(3, 5),
                                min_df=2,
                                max_features=20000,
                                sublinear_tf=True,
                            ),
                        ),
                    ]
                ),
            ),
            ("lr", LogisticRegression(C=2.0, max_iter=300, solver="liblinear")),
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
