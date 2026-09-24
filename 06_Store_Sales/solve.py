#!/usr/bin/env python3
"""Store Sales – LightGBM with calendar, promo, holiday, and lag>=16."""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "data" / "store-sales"
LAGS = (16, 21, 28, 35)
CATS = ["store_nbr", "family", "type", "cluster"]
HORIZON_START = pd.Timestamp("2017-07-31")


def rmsle(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 0, None)
    return float(np.sqrt(np.mean((np.log1p(p) - np.log1p(y)) ** 2)))


def load_frame() -> pd.DataFrame:
    train = pd.read_csv(DATA / "train.csv", parse_dates=["date"])
    test = pd.read_csv(DATA / "test.csv", parse_dates=["date"])
    train["sales"] = train["sales"].astype(np.float32)
    test["sales"] = np.nan
    both = pd.concat([train, test], ignore_index=True)

    stores = pd.read_csv(DATA / "stores.csv")
    both = both.merge(stores[["store_nbr", "type", "cluster"]], on="store_nbr", how="left")

    holidays = pd.read_csv(DATA / "holidays_events.csv", parse_dates=["date"])
    holidays = holidays.loc[(~holidays["transferred"]) & (holidays["type"] != "Work Day")]
    national = set(holidays.loc[holidays["locale"] == "National", "date"])
    both["is_holiday"] = both["date"].isin(national).astype(np.int8)
    both["dow"] = both["date"].dt.dayofweek.astype(np.int8)
    both["day"] = both["date"].dt.day.astype(np.int8)
    both["month"] = both["date"].dt.month.astype(np.int8)
    both["is_weekend"] = (both["dow"] >= 5).astype(np.int8)

    both = both.sort_values(["store_nbr", "family", "date"])
    grouped = both.groupby(["store_nbr", "family"], sort=False)["sales"]
    for lag in LAGS:
        both[f"lag{lag}"] = grouped.shift(lag)
    both["lag_mean"] = both[[f"lag{lag}" for lag in LAGS]].mean(axis=1)
    return both


def matrix(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "store_nbr",
        "family",
        "type",
        "cluster",
        "onpromotion",
        "is_holiday",
        "dow",
        "day",
        "month",
        "is_weekend",
        "lag_mean",
        *[f"lag{lag}" for lag in LAGS],
    ]
    out = df[cols].copy()
    for col in CATS:
        out[col] = out[col].astype("category")
    return out


def fit_model(x: pd.DataFrame, y: np.ndarray) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.06,
        num_leaves=63,
        min_child_samples=40,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )
    model.fit(x, np.log1p(y), categorical_feature=CATS)
    return model


def main() -> None:
    frame = load_frame()
    train = frame.loc[frame["sales"].notna() & (frame["date"] >= "2016-01-01")].copy()
    valid = train["date"] >= HORIZON_START
    # lag>=16 only looks at dates before the 16-day holdout, so this RMSLE matches the test horizon.
    model = fit_model(matrix(train.loc[~valid]), train.loc[~valid, "sales"].to_numpy())
    pred_valid = np.expm1(model.predict(matrix(train.loc[valid]))).clip(min=0)
    print(f"holdout RMSLE (2017-07-31..08-15): {rmsle(train.loc[valid, 'sales'].to_numpy(), pred_valid):.4f}")

    model = fit_model(matrix(train), train["sales"].to_numpy())
    test = frame.loc[frame["sales"].isna()].sort_values("id")
    sales = np.expm1(model.predict(matrix(test))).clip(min=0)
    out = pd.DataFrame({"id": test["id"].astype(int), "sales": sales})
    path = ROOT / "submission.csv"
    out.to_csv(path, index=False)
    print(f"wrote {path}  rows={len(out)}  pred_mean={sales.mean():.3f}")


if __name__ == "__main__":
    main()
