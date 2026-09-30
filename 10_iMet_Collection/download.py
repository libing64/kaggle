#!/usr/bin/env python3
"""Download iMet Collection 2019 (FGVC6) into 10_iMet_Collection/data.

Retries automatically on Kaggle 429 rate limits.
"""

from __future__ import annotations

import time
import zipfile
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi
from requests import HTTPError

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
COMP = "imet-2019-fgvc6"


def download_with_retry() -> Path:
    DATA.mkdir(parents=True, exist_ok=True)
    zip_path = DATA / f"{COMP}.zip"
    # Competition ships train.csv + train/ + test/ (+ labels.csv).
    if (DATA / "train.csv").exists() and (DATA / "train").is_dir() and (DATA / "test").is_dir():
        print(f"already present: {DATA}")
        return DATA

    while True:
        api = KaggleApi()
        api.authenticate()
        try:
            print(f"downloading {COMP} (~22.6GB) -> {DATA}", flush=True)
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
        print(f"unzipping {zip_path}", flush=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(DATA)
        print("unzip done", flush=True)
    return DATA


if __name__ == "__main__":
    download_with_retry()
    print("done")
