#!/usr/bin/env python3
"""Download Lyft data then train/infer with the lyft conda env."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> None:
    env = dict(**os.environ, PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="python")
    py = sys.executable
    subprocess.check_call([py, str(ROOT / "download.py")])
    # Prefer current interpreter if l5kit imports; else fall back to lyft env.
    try:
        subprocess.check_call(
            [py, "-c", "import l5kit"],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        subprocess.check_call([py, "-u", str(ROOT / "solve.py")], env=env)
        return
    except subprocess.CalledProcessError:
        pass
    lyft_py = Path.home() / ".conda" / "envs" / "lyft" / "bin" / "python"
    if lyft_py.exists():
        subprocess.check_call([str(lyft_py), "-u", str(ROOT / "solve.py")], env=env)
        return
    raise SystemExit("l5kit not importable; see README.md")


if __name__ == "__main__":
    main()
