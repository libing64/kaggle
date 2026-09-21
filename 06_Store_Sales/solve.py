#!/usr/bin/env python3
"""Store Sales – naive seasonal mean (store, family, weekday)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "data" / "store-sales"


def main() -> None:
    train = pd.read_csv(
        DATA / "train.csv",
        usecols=["date", "store_nbr", "family", "sales"],
        parse_dates=["date"],
    )
    test = pd.read_csv(DATA / "test.csv", parse_dates=["date"])

    # Recency window: last 8 weeks of train (ends 2017-08-15).
    recent = train[train["date"] >= "2017-06-21"].copy()
    recent["dow"] = recent["date"].dt.dayofweek
    test = test.copy()
    test["dow"] = test["date"].dt.dayofweek

    by_dow = recent.groupby(["store_nbr", "family", "dow"], as_index=False)["sales"].mean()
    by_sf = recent.groupby(["store_nbr", "family"], as_index=False)["sales"].mean()
    global_mean = float(recent["sales"].mean())

    pred = test.merge(by_dow, on=["store_nbr", "family", "dow"], how="left")
    pred = pred.merge(by_sf, on=["store_nbr", "family"], how="left", suffixes=("", "_sf"))
    sales = pred["sales"].fillna(pred["sales_sf"]).fillna(global_mean).clip(lower=0)

    out = pd.DataFrame({"id": test["id"], "sales": sales})
    path = ROOT / "submission.csv"
    out.to_csv(path, index=False)
    print(
        "naive mean by (store, family, weekday) over last 8 weeks; "
        f"fallback=(store, family) then global={global_mean:.3f}"
    )
    print(f"wrote {path}  rows={len(out)}  pred_mean={sales.mean():.3f}")


if __name__ == "__main__":
    main()
