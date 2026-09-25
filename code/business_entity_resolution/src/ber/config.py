"""Paths shared by every pipeline stage.

DATA_DIR points at the challenge `dataset/` folder (train/ and test/ inside).
WORK_DIR holds intermediate parquet files and models; OUTPUT_DIR the two TSVs.
All three can be overridden with environment variables.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("BER_DATA_DIR", ROOT.parents[1] / "data"))
WORK_DIR = Path(os.environ.get("BER_WORK_DIR", ROOT.parents[1] / "work"))
OUTPUT_DIR = Path(os.environ.get("BER_OUTPUT_DIR", ROOT.parents[1] / "output"))

N_JOBS = int(os.environ.get("BER_JOBS", os.cpu_count() or 4))


def raw_path(split: str, src: int) -> Path:
    return DATA_DIR / split / f"{split}_source{src}.tsv"


def work(*parts) -> Path:
    p = WORK_DIR.joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
