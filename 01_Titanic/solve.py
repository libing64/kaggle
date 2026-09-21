#!/usr/bin/env python3
"""Titanic – Machine Learning from Disaster

Covers: missing values, categorical encoding, feature engineering,
cross-validation, Logistic Regression / Random Forest vs a naive MLP.

Run (needs pandas / numpy / scikit-learn):
    python 01_Titanic/solve.py
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from mlp import NaiveMLP
else:
    from .mlp import NaiveMLP

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "rn2019b-titanic"
OUT_DIR = ROOT / "outputs"

TITLE_MAP = {
    "Mr": "Mr",
    "Mrs": "Mrs",
    "Miss": "Miss",
    "Master": "Master",
    "Ms": "Miss",
    "Mlle": "Miss",
    "Mme": "Mrs",
}


def extract_title(name: str) -> str:
    match = re.search(r" ([A-Za-z]+)\.", str(name))
    if not match:
        return "Rare"
    return TITLE_MAP.get(match.group(1), "Rare")


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Title"] = out["Name"].map(extract_title)
    out["FamilySize"] = out["SibSp"] + out["Parch"] + 1
    out["IsAlone"] = (out["FamilySize"] == 1).astype(int)
    out["SmallFamily"] = ((out["FamilySize"] >= 2) & (out["FamilySize"] <= 4)).astype(int)
    out["HasCabin"] = out["Cabin"].notna().astype(int)
    deck = out["Cabin"].astype(str).str[0]
    out["Deck"] = deck.where(out["Cabin"].notna() & deck.ne("T") & deck.ne("n"), "U")
    return out


def fit_imputers(train: pd.DataFrame) -> dict:
    fare = train["Fare"].fillna(train.groupby("Pclass")["Fare"].transform("median"))
    _, fare_bins = pd.qcut(fare.clip(lower=0), q=4, retbins=True, duplicates="drop")
    fare_bins[0] = -np.inf
    fare_bins[-1] = np.inf
    return {
        "age_by_title_pclass": train.groupby(["Title", "Pclass"])["Age"].median(),
        "age_by_title": train.groupby("Title")["Age"].median(),
        "age_global": train["Age"].median(),
        "fare_by_pclass": train.groupby("Pclass")["Fare"].median(),
        "fare_global": train["Fare"].median(),
        "embarked_mode": train["Embarked"].mode().iloc[0],
        "ticket_counts": train["Ticket"].value_counts(),
        "fare_bins": fare_bins,
    }


def apply_imputers(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    out = df.copy()
    age_keys = list(zip(out["Title"], out["Pclass"]))
    age = out["Age"].copy()
    mapped = pd.Series(age_keys, index=out.index).map(stats["age_by_title_pclass"])
    mapped = mapped.fillna(out["Title"].map(stats["age_by_title"])).fillna(stats["age_global"])
    out["Age"] = age.fillna(mapped)

    fare = out["Fare"].copy()
    fare_mapped = out["Pclass"].map(stats["fare_by_pclass"]).fillna(stats["fare_global"])
    out["Fare"] = fare.fillna(fare_mapped)
    out["Embarked"] = out["Embarked"].fillna(stats["embarked_mode"])

    ticket_freq = out["Ticket"].map(stats["ticket_counts"]).fillna(1).astype(int)
    out["TicketFreq"] = ticket_freq
    out["FarePerPerson"] = out["Fare"] / out["FamilySize"].clip(lower=1)
    out["AgeBin"] = pd.cut(
        out["Age"],
        bins=[0, 12, 18, 25, 35, 50, 80],
        labels=["child", "teen", "young", "adult", "mid", "senior"],
        include_lowest=True,
    ).astype(str)
    n_fare_bins = len(stats["fare_bins"]) - 1
    fare_labels = [f"q{i}" for i in range(1, n_fare_bins + 1)]
    out["FareBin"] = pd.cut(out["Fare"].clip(lower=0), bins=stats["fare_bins"], labels=fare_labels).astype(str)
    return out


NUM_COLS = [
    "Age",
    "Fare",
    "FamilySize",
    "FarePerPerson",
    "TicketFreq",
    "SibSp",
    "Parch",
    "IsAlone",
    "SmallFamily",
    "HasCabin",
]
CAT_COLS = ["Pclass", "Sex", "Embarked", "Title", "Deck", "AgeBin", "FareBin"]


def make_preprocessor(scale: bool) -> ColumnTransformer:
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            *([("scaler", StandardScaler())] if scale else []),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric, NUM_COLS),
            ("cat", categorical, CAT_COLS),
        ]
    )


def make_models() -> dict[str, Pipeline]:
    lr = Pipeline(
        steps=[
            ("prep", make_preprocessor(scale=True)),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    C=1.0,
                    solver="lbfgs",
                    random_state=42,
                ),
            ),
        ]
    )
    rf = Pipeline(
        steps=[
            ("prep", make_preprocessor(scale=False)),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=400,
                    max_depth=6,
                    min_samples_split=8,
                    min_samples_leaf=3,
                    max_features="sqrt",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    mlp = Pipeline(
        steps=[
            ("prep", make_preprocessor(scale=True)),
            (
                "clf",
                NaiveMLP(
                    hidden_layer_sizes=(32, 16),
                    learning_rate=0.05,
                    epochs=200,
                    batch_size=32,
                    l2=1e-4,
                    random_state=42,
                ),
            ),
        ]
    )
    return {"logistic_regression": lr, "random_forest": rf, "naive_mlp": mlp}


def summarize(train: pd.DataFrame, y: pd.Series) -> None:
    print("=" * 60)
    print("Data overview")
    print("=" * 60)
    print(f"train rows: {len(train)}  test-ready features: {train.shape[1]}")
    print("missing rates:")
    miss = train.isna().mean().sort_values(ascending=False)
    print(miss[miss > 0].to_string())
    print("\nsurvival rate: {:.3f}".format(y.mean()))
    print("survival by Sex:")
    tmp = train.assign(Survived=y.values)
    print(tmp.groupby("Sex")["Survived"].mean().to_string())
    print("survival by Pclass:")
    print(tmp.groupby("Pclass")["Survived"].mean().to_string())


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    x_train = pd.read_csv(DATA_DIR / "X_train.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv")
    x_test = pd.read_csv(DATA_DIR / "X_test.csv")

    train = x_train.merge(y_train, on="PassengerId", how="inner")
    if len(train) != len(x_train):
        raise SystemExit("X_train / y_train PassengerId mismatch")
    y = train["Survived"].astype(int)
    summarize(train, y)

    train_fe = add_features(train)
    test_fe = add_features(x_test)
    stats = fit_imputers(train_fe)
    train_fe = apply_imputers(train_fe, stats)
    test_fe = apply_imputers(test_fe, stats)

    X = train_fe[NUM_COLS + CAT_COLS]
    X_test = test_fe[NUM_COLS + CAT_COLS]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models = make_models()
    cv_means: dict[str, float] = {}
    oof_proba: dict[str, np.ndarray] = {}

    print("\n" + "=" * 60)
    print("5-fold stratified CV (Accuracy)")
    print("=" * 60)
    for name, model in models.items():
        # NaiveMLP is a Python SGD loop; keep CV in-process to avoid pickle overhead.
        n_jobs = 1 if name == "naive_mlp" else -1
        scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=n_jobs)
        cv_means[name] = float(scores.mean())
        oof_proba[name] = cross_val_predict(model, X, y, cv=cv, method="predict_proba", n_jobs=n_jobs)[:, 1]
        print(f"{name:22s}  mean={scores.mean():.4f}  std={scores.std():.4f}  folds={np.round(scores, 4)}")

    blend_oof = 0.45 * oof_proba["logistic_regression"] + 0.55 * oof_proba["random_forest"]
    blend_acc = accuracy_score(y, (blend_oof >= 0.5).astype(int))
    cv_means["blend_lr_rf"] = blend_acc
    print(f"{'blend_lr_rf':22s}  oof_acc={blend_acc:.4f}  (0.45*LR + 0.55*RF @ 0.5)")

    classic = {k: v for k, v in cv_means.items() if k != "naive_mlp"}
    best_name = max(classic, key=classic.get)
    print(f"\nselected (tabular): {best_name}  score={cv_means[best_name]:.4f}")
    print(f"naive_mlp (compare): {cv_means['naive_mlp']:.4f}")

    for name, model in models.items():
        model.fit(X, y)

    lr_proba = models["logistic_regression"].predict_proba(X_test)[:, 1]
    rf_proba = models["random_forest"].predict_proba(X_test)[:, 1]
    blend_proba = 0.45 * lr_proba + 0.55 * rf_proba
    mlp_pred = models["naive_mlp"].predict(X_test)

    if best_name == "blend_lr_rf":
        pred = (blend_proba >= 0.5).astype(int)
    elif best_name == "logistic_regression":
        pred = models["logistic_regression"].predict(X_test)
    else:
        pred = models["random_forest"].predict(X_test)

    submission = pd.DataFrame({"PassengerId": x_test["PassengerId"], "Survived": pred.astype(int)})
    out_path = ROOT / "submission.csv"
    submission.to_csv(out_path, index=False)
    submission.to_csv(OUT_DIR / "submission.csv", index=False)

    mlp_submission = pd.DataFrame({"PassengerId": x_test["PassengerId"], "Survived": mlp_pred.astype(int)})
    mlp_path = ROOT / "submission_mlp.csv"
    mlp_submission.to_csv(mlp_path, index=False)
    mlp_submission.to_csv(OUT_DIR / "submission_mlp.csv", index=False)

    report = pd.DataFrame(
        [{"model": k, "cv_accuracy": v} for k, v in cv_means.items()]
    ).sort_values("cv_accuracy", ascending=False)
    report.to_csv(OUT_DIR / "cv_scores.csv", index=False)

    print("\n" + "=" * 60)
    print(f"wrote {out_path}")
    print(f"wrote {mlp_path}")
    print(f"predicted survival rate: {submission['Survived'].mean():.3f}")
    print(f"mlp predicted survival rate: {mlp_submission['Survived'].mean():.3f}")
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
