#!/usr/bin/env python3
"""Download Cassava Leaf Disease Classification data.

Retries automatically when Kaggle returns 429.
"""

from __future__ import annotations

import time
import zipfile
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi
from requests import HTTPError

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "data" / "cassava"
COMP = "cassava-leaf-disease-classification"


def download_with_retry() -> Path:
    DATA.mkdir(parents=True, exist_ok=True)
    zip_path = DATA / f"{COMP}.zip"
    if (DATA / "train.csv").exists() and (DATA / "train_images").is_dir():
        print(f"already present: {DATA}")
        return DATA

    while True:
        api = KaggleApi()
        api.authenticate()
        try:
            print(f"downloading {COMP} -> {DATA}", flush=True)
            api.competition_download_files(COMP, path=str(DATA), force=False, quiet=False)
            break
        except HTTPError as exc:
            if exc.response is None or exc.response.status_code != 429:
                raise
            wait = int(exc.response.headers.get("retry-after", "3600"))
            wait = max(wait + 30, 60)
            print(f"rate limited; sleeping {wait}s (~{wait / 3600:.1f}h)", flush=True)
            time.sleep(wait)

    if zip_path.exists():
        print(f"unzipping {zip_path}")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(DATA)
    return DATA


if __name__ == "__main__":
    download_with_retry()
    print("done")
