#!/usr/bin/env python3
"""Spaceship Titanic – naive RandomForest after light feature splits."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "data" / "spaceship-titanic"


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    cabin = out["Cabin"].fillna("//").str.split("/", expand=True)
    out["Deck"] = cabin[0].replace("", np.nan)
    out["CabinNum"] = pd.to_numeric(cabin[1], errors="coerce")
    out["Side"] = cabin[2].replace("", np.nan)
    out["Group"] = out["PassengerId"].str.split("_").str[0]
    out["GroupSize"] = out.groupby("Group")["PassengerId"].transform("count")
    spend = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    out["Spend"] = out[spend].sum(axis=1)
    out["CryoSleep"] = out["CryoSleep"].map({True: 1, False: 0, "True": 1, "False": 0})
    out["VIP"] = out["VIP"].map({True: 1, False: 0, "True": 1, "False": 0})
    return out


def main() -> None:
    train = add_features(pd.read_csv(DATA / "train.csv"))
    test = add_features(pd.read_csv(DATA / "test.csv"))
    y = train["Transported"].astype(int)

    num_cols = ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck", "Spend", "GroupSize", "CabinNum", "CryoSleep", "VIP"]
    cat_cols = ["HomePlanet", "Destination", "Deck", "Side"]
    X = train[num_cols + cat_cols]
    X_test = test[num_cols + cat_cols]

    model = Pipeline(
        [
            (
                "prep",
                ColumnTransformer(
                    [
                        ("num", SimpleImputer(strategy="median"), num_cols),
                        (
                            "cat",
                            Pipeline(
                                [
                                    ("imputer", SimpleImputer(strategy="most_frequent")),
                                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                                ]
                            ),
                            cat_cols,
                        ),
                    ]
                ),
            ),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=8,
                    min_samples_leaf=4,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    acc = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    print(f"5-fold Accuracy: mean={acc.mean():.4f}  std={acc.std():.4f}  folds={np.round(acc, 4)}")

    model.fit(X, y)
    pred = model.predict(X_test).astype(bool)
    out = pd.DataFrame({"PassengerId": test["PassengerId"], "Transported": pred})
    path = ROOT / "submission.csv"
    out.to_csv(path, index=False)
    print(f"wrote {path}  rows={len(out)}")


if __name__ == "__main__":
    main()
