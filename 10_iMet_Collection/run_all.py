#!/usr/bin/env python3
"""Download iMet data then run teacher→student distillation."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> None:
    py = sys.executable
    subprocess.check_call([py, str(ROOT / "download.py")])
    subprocess.check_call([py, "-u", str(ROOT / "solve.py")])


if __name__ == "__main__":
    main()
