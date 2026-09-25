"""Score the test candidates with the stage B fold models and write the submission."""
import json
import sys

import lightgbm as lgb
import numpy as np
import polars as pl

from . import decide, output
from . import features as F
from . import stage_b as B
from .config import N_JOBS, OUTPUT_DIR, work


def score(split: str = "test", tag: str = "main", k: int = 3) -> pl.DataFrame:
    feats = json.loads(work("models", f"stage_b_{tag}_features.json").read_text())
    models = [lgb.Booster(model_file=str(work("models", f"stage_b_{tag}_fold{f}.txt"))) for f in range(k)]
    out = []
    for p in B.parts(split):
        d = pl.read_parquet(p, columns=["qr", "tr"] + feats)
        X = d.select(feats).to_numpy().astype(np.float32)
        pr = np.mean([m.predict(X, num_threads=N_JOBS) for m in models], axis=0)
        out.append(d.select("qr", "tr").with_columns(pl.Series("p", pr.astype(np.float32))))
    S = pl.concat(out)
    S.write_parquet(work(f"{split}_scores_{tag}.parquet"))
    return S


def run(tag: str = "main", rule: str = "expected_f", out_dir=OUTPUT_DIR):
    S = score("test", tag)
    R = F.load_records("test").select("id", "src", "country")
    cands = pl.read_parquet(work("test_cands.parquet")).select("qr", "tr")
    S1 = decide.one_per_target(S)
    M = decide.select_expected_f(S1) if rule == "expected_f" else decide.select_threshold(S1, float(rule))
    output.write(R, cands, M.select("qr", "tr"), out_dir=out_dir)
    return S, M


if __name__ == "__main__":
    run(*(sys.argv[1:2] or ["main"]))
