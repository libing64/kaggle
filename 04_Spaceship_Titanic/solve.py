#!/usr/bin/env python3
"""Spaceship Titanic – group-aware fills, log spend, LightGBM."""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "data" / "spaceship-titanic"
SPEND = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
CATS = ["HomePlanet", "Destination", "Deck", "Side"]
NUMS = [
    "Age",
    "CryoSleep",
    "VIP",
    "CabinNum",
    "GroupSize",
    "IsAlone",
    "Spend",
    "LuxSpend",
    *SPEND,
]


def _group_mode(frame: pd.DataFrame, col: str) -> pd.Series:
    mode = frame.groupby("Group")[col].agg(lambda s: s.dropna().mode().iloc[0] if s.notna().any() else np.nan)
    return frame["Group"].map(mode)


def add_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = train.copy()
    test = test.copy()
    train["_src"] = "train"
    test["_src"] = "test"
    both = pd.concat([train, test], ignore_index=True)
    both["Group"] = both["PassengerId"].str.split("_").str[0]
    both["GroupSize"] = both.groupby("Group")["PassengerId"].transform("count")
    both["IsAlone"] = (both["GroupSize"] == 1).astype(int)

    cabin = both["Cabin"].fillna("//").str.split("/", expand=True)
    both["Deck"] = cabin[0].replace("", np.nan)
    both["CabinNum"] = pd.to_numeric(cabin[1], errors="coerce")
    both["Side"] = cabin[2].replace("", np.nan)
    both["Deck"] = both["Deck"].fillna(_group_mode(both, "Deck"))
    both["Side"] = both["Side"].fillna(_group_mode(both, "Side"))
    both["HomePlanet"] = both["HomePlanet"].fillna(_group_mode(both, "HomePlanet"))

    for col in SPEND:
        both[col] = both[col].fillna(0.0)
    both["Spend"] = both[SPEND].sum(axis=1)
    both["LuxSpend"] = both["Spa"] + both["VRDeck"] + both["RoomService"]
    both.loc[both["Spend"] > 0, "CryoSleep"] = False
    both.loc[(both["Spend"] == 0) & both["CryoSleep"].isna(), "CryoSleep"] = True
    both["CryoSleep"] = both["CryoSleep"].map({True: 1, False: 0, "True": 1, "False": 0}).fillna(0)
    both["VIP"] = both["VIP"].map({True: 1, False: 0, "True": 1, "False": 0})
    both["Age"] = both["Age"].fillna(both.groupby("HomePlanet")["Age"].transform("median"))
    for col in SPEND + ["Spend", "LuxSpend"]:
        both[col] = np.log1p(both[col])

    train_fe = both.loc[both["_src"] == "train"].drop(columns="_src")
    test_fe = both.loc[both["_src"] == "test"].drop(columns="_src")
    return train_fe, test_fe


def main() -> None:
    train, test = add_features(pd.read_csv(DATA / "train.csv"), pd.read_csv(DATA / "test.csv"))
    y = train["Transported"].astype(int).to_numpy()
    cols = NUMS + CATS
    X = train[cols].copy()
    X_test = test[cols].copy()
    for col in CATS:
        levels = pd.concat([X[col], X_test[col]], ignore_index=True).astype("category")
        cat = levels.dtype
        X[col] = pd.Categorical(X[col], categories=cat.categories)
        X_test[col] = pd.Categorical(X_test[col], categories=cat.categories)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for tr, va in cv.split(X, y):
        model = lgb.LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
        )
        model.fit(X.iloc[tr], y[tr], categorical_feature=CATS)
        pred = model.predict(X.iloc[va])
        scores.append(float((pred == y[va]).mean()))
    scores_arr = np.array(scores)
    print(f"5-fold Accuracy: mean={scores_arr.mean():.4f}  std={scores_arr.std():.4f}  folds={np.round(scores_arr, 4)}")

    model = lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )
    model.fit(X, y, categorical_feature=CATS)
    pred = model.predict(X_test).astype(bool)
    out = pd.DataFrame({"PassengerId": test["PassengerId"], "Transported": pred})
    path = ROOT / "submission.csv"
    out.to_csv(path, index=False)
    print(f"wrote {path}  rows={len(out)}")


if __name__ == "__main__":
    main()
