#!/usr/bin/env python3
"""Download Lyft Motion Prediction data into 11_lyft_motion_prediction/data.

Retries on Kaggle 429. Full competition zip ≈ 18.3GB.
"""

from __future__ import annotations

import time
import zipfile
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi
from requests import HTTPError

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
COMP = "lyft-motion-prediction-autonomous-vehicles"


def _ready(data: Path) -> bool:
    # Competition layout: scenes/*.zarr (+ maps + sample_submission).
    scenes = data / "scenes"
    if scenes.is_dir() and any(scenes.glob("*.zarr")):
        return True
    # Sometimes zarr dirs sit at data root.
    return any(data.glob("*.zarr")) or (data / "sample.zarr").exists()


def download_with_retry() -> Path:
    DATA.mkdir(parents=True, exist_ok=True)
    zip_path = DATA / f"{COMP}.zip"
    if _ready(DATA):
        print(f"already present: {DATA}")
        return DATA

    while True:
        api = KaggleApi()
        api.authenticate()
        try:
            print(f"downloading {COMP} (~18.3GB) -> {DATA}", flush=True)
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
        # Free space after successful extract.
        zip_path.unlink(missing_ok=True)
        print(f"removed {zip_path.name}", flush=True)
    return DATA


if __name__ == "__main__":
    download_with_retry()
    print("done")
