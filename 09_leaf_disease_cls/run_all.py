#!/usr/bin/env python3
"""Wait out Kaggle 429, then download Cassava data and train."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable


def main() -> None:
    print("step 1/2 download", flush=True)
    subprocess.check_call([PY, "-u", str(ROOT / "download.py")])
    print("step 2/2 train", flush=True)
    subprocess.check_call([PY, "-u", str(ROOT / "solve.py")])


if __name__ == "__main__":
    main()
