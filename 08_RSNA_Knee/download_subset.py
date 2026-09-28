#!/usr/bin/env python3
"""Download CSVs are already local. Fetch DICOMs for labeled studies and the public test.

The full archive is about 570 GB. This keeps five slices from the fluid-sensitive
sagittal and coronal series of the 58 labeled studies, plus the public test series.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.competitions.types.competition_api_service import ApiListDataTreeFilesRequest

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "data" / "rsna-knee"
COMP = "rsna-knee-abnormality-detection"
LABELS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]


def list_dir(api: KaggleApi, path: str) -> tuple[list[str], list[str]]:
    dirs, files = [], []
    token = None
    with api.build_kaggle_client() as kaggle:
        while True:
            req = ApiListDataTreeFilesRequest()
            req.competition_name = COMP
            req.path = path
            req.page_size = 200
            if token:
                req.page_token = token
            resp = kaggle.competitions.competition_api_client.list_data_tree_files(req)
            dirs.extend(item.relative_url for item in (resp.directories or []) if item.relative_url)
            files.extend(item.name for item in (resp.files or []) if item.name)
            token = resp.next_page_token or ""
            if not token:
                break
    return dirs, files


def sample_slices(files: list[str]) -> list[str]:
    files = sorted(files)
    if not files:
        return []
    indexes = {0, len(files) // 4, len(files) // 2, 3 * len(files) // 4, len(files) - 1}
    return [files[index] for index in sorted(indexes)]


def pick_series(group: pd.DataFrame) -> list[str]:
    chosen = []
    for plane in ("Sagittal", "Coronal"):
        subset = group[group["Anatomical_Plane"] == plane]
        fluid = subset[subset["Fluid_Sensitive"] == 1]
        pool = fluid if len(fluid) else subset
        if len(pool):
            chosen.append(pool.iloc[0]["SeriesInstanceUID"])
    if not chosen:
        chosen.append(group.iloc[0]["SeriesInstanceUID"])
    return chosen


def download_one(file_name: str, dest: Path) -> str:
    import time

    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return "skip"
    delay = 2.0
    for _ in range(6):
        try:
            api = KaggleApi()
            api.authenticate()
            api.competition_download_file(COMP, file_name, path=str(dest.parent), quiet=True)
            return "ok"
        except requests.HTTPError as exc:
            if exc.response is None or exc.response.status_code != 429:
                raise
            time.sleep(delay)
            delay *= 2
    raise RuntimeError(f"rate limited: {file_name}")


def main() -> None:
    train = pd.read_csv(DATA / "train.csv")
    series = pd.read_csv(DATA / "train_series.csv")
    test_series = pd.read_csv(DATA / "test_series.csv")
    labeled_ids = set(train.dropna(subset=LABELS)["StudyInstanceUID"])
    jobs: list[tuple[str, Path]] = []
    api = KaggleApi()
    api.authenticate()

    wanted = series[series["StudyInstanceUID"].isin(labeled_ids)]
    for study_id, group in wanted.groupby("StudyInstanceUID"):
        for series_id in pick_series(group):
            remote = f"train_series/{study_id}/{series_id}"
            _, files = list_dir(api, remote)
            for name in sample_slices(files):
                jobs.append((f"{remote}/{name}", DATA / "train_series" / study_id / series_id / name))

    for row in test_series.itertuples(index=False):
        remote = f"test_series/{row.StudyInstanceUID}/{row.SeriesInstanceUID}"
        _, files = list_dir(api, remote)
        for name in sample_slices(files):
            jobs.append((f"{remote}/{name}", DATA / "test_series" / row.StudyInstanceUID / row.SeriesInstanceUID / name))

    print(f"files to fetch: {len(jobs)}")
    done = 0
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(download_one, remote, dest) for remote, dest in jobs]
        for fut in as_completed(futures):
            fut.result()
            done += 1
            if done % 100 == 0 or done == len(jobs):
                print(f"{done}/{len(jobs)}")


if __name__ == "__main__":
    main()
