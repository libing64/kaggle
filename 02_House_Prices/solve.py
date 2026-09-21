#!/usr/bin/env python3
"""House Prices – naive Ridge on log1p(SalePrice)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "data" / "house-prices"


def main() -> None:
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = np.log1p(train["SalePrice"])
    X = train.drop(columns=["Id", "SalePrice"])
    X_test = test.drop(columns=["Id"])

    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()

    model = Pipeline(
        [
            (
                "prep",
                ColumnTransformer(
                    [
                        (
                            "num",
                            Pipeline(
                                [
                                    ("imputer", SimpleImputer(strategy="median")),
                                    ("scaler", StandardScaler()),
                                ]
                            ),
                            num_cols,
                        ),
                        (
                            "cat",
                            Pipeline(
                                [
                                    ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
                                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                                ]
                            ),
                            cat_cols,
                        ),
                    ]
                ),
            ),
            ("ridge", Ridge(alpha=10.0)),
        ]
    )

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse = -cross_val_score(model, X, y, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1)
    print(f"5-fold log-RMSE: mean={rmse.mean():.4f}  std={rmse.std():.4f}  folds={np.round(rmse, 4)}")

    model.fit(X, y)
    pred = np.expm1(model.predict(X_test)).clip(min=0)
    out = pd.DataFrame({"Id": test["Id"], "SalePrice": pred})
    path = ROOT / "submission.csv"
    out.to_csv(path, index=False)
    print(f"wrote {path}  rows={len(out)}")


if __name__ == "__main__":
    main()
