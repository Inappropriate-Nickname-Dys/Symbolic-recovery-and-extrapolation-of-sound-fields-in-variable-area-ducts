"""Resolve code and output locations without depending on the archived project."""

import os
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = Path(os.environ.get("ACOUSTIC_REPOSITORY_ROOT", SOURCE_ROOT.parents[1])).resolve()
ROOT = Path(os.environ.get("ACOUSTIC_OUTPUT_ROOT", REPOSITORY_ROOT / "runs" / "full_v1")).resolve()
