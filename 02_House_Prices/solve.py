#!/usr/bin/env python3
"""House Prices – ordinal features, size aggregates, drop outliers, Ridge."""

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

QUAL = {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
FIN = {"Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6}
ORDINAL = {
    "ExterQual": QUAL,
    "ExterCond": QUAL,
    "BsmtQual": QUAL,
    "BsmtCond": QUAL,
    "HeatingQC": QUAL,
    "KitchenQual": QUAL,
    "FireplaceQu": QUAL,
    "GarageQual": QUAL,
    "GarageCond": QUAL,
    "PoolQC": QUAL,
    "BsmtExposure": {"No": 1, "Mn": 2, "Av": 3, "Gd": 4},
    "BsmtFinType1": FIN,
    "BsmtFinType2": FIN,
    "GarageFinish": {"Unf": 1, "RFn": 2, "Fin": 3},
    "Functional": {"Sal": 1, "Sev": 2, "Maj2": 3, "Maj1": 4, "Mod": 5, "Min2": 6, "Min1": 7, "Typ": 8},
    "Fence": {"MnWw": 1, "GdWo": 2, "MnPrv": 3, "GdPrv": 4},
    "LotShape": {"IR3": 1, "IR2": 2, "IR1": 3, "Reg": 4},
    "LandSlope": {"Sev": 1, "Mod": 2, "Gtl": 3},
    "PavedDrive": {"N": 0, "P": 1, "Y": 2},
}


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["MSSubClass"] = out["MSSubClass"].astype(str)
    for col, mapping in ORDINAL.items():
        out[col] = out[col].map(mapping).fillna(0)
    out["TotalSF"] = out["TotalBsmtSF"].fillna(0) + out["1stFlrSF"].fillna(0) + out["2ndFlrSF"].fillna(0)
    out["TotalBath"] = (
        out["FullBath"].fillna(0)
        + 0.5 * out["HalfBath"].fillna(0)
        + out["BsmtFullBath"].fillna(0)
        + 0.5 * out["BsmtHalfBath"].fillna(0)
    )
    out["HouseAge"] = out["YrSold"] - out["YearBuilt"]
    out["RemodAge"] = out["YrSold"] - out["YearRemodAdd"]
    out["HasGarage"] = (out["GarageArea"].fillna(0) > 0).astype(int)
    return out


def log_skewed(train: pd.DataFrame, test: pd.DataFrame, skip: set[str]) -> None:
    num = train.select_dtypes(include=np.number).columns
    for col in num:
        if col in skip:
            continue
        if train[col].min() < 0:
            continue
        if train[col].skew() > 0.75:
            train[col] = np.log1p(train[col].clip(lower=0))
            test[col] = np.log1p(test[col].clip(lower=0))


def main() -> None:
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    # Two partial-sale outliers that dominate one CV fold.
    train = train.loc[~((train["GrLivArea"] > 4000) & (train["SalePrice"] < 300000))].copy()

    y = np.log1p(train["SalePrice"])
    train_x = engineer(train.drop(columns=["Id", "SalePrice"]))
    test_x = engineer(test.drop(columns=["Id"]))
    log_skewed(train_x, test_x, skip=set(ORDINAL) | {"HasGarage", "TotalBath"})

    num_cols = train_x.select_dtypes(include=np.number).columns.tolist()
    cat_cols = train_x.select_dtypes(exclude=np.number).columns.tolist()
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
                                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                                ]
                            ),
                            cat_cols,
                        ),
                    ]
                ),
            ),
            ("ridge", Ridge(alpha=12.0)),
        ]
    )

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse = -cross_val_score(model, train_x, y, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1)
    print(f"5-fold log-RMSE: mean={rmse.mean():.4f}  std={rmse.std():.4f}  folds={np.round(rmse, 4)}")

    model.fit(train_x, y)
    pred = np.expm1(model.predict(test_x)).clip(min=0)
    out = pd.DataFrame({"Id": test["Id"], "SalePrice": pred})
    path = ROOT / "submission.csv"
    out.to_csv(path, index=False)
    print(f"wrote {path}  rows={len(out)}")


if __name__ == "__main__":
    main()
