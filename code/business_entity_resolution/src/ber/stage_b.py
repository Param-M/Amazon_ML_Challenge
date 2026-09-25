"""Stage B: the final pairwise matcher.

build  : rich features for every stage-A candidate (train and test)
train  : LightGBM, K folds grouped by Source 1 record -> out-of-fold scores on
         train (used to tune the decision rule) + fold models (used on test)
"""
import json
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from . import features as F
from . import rich as RF
from . import stats
from .config import N_JOBS, work

EXCLUDE = {"qr", "tr", "y", "fold", "country_id"}


def build(split: str, limit_q: int | None = None):
    t0 = time.time()
    R = F.load_records(split)
    T = stats.load(split)
    C = pl.read_parquet(work(f"{split}_cands.parquet"))
    if limit_q is not None:
        C = C.filter(pl.col("qr").is_in(C["qr"].unique().sort().head(limit_q).implode()))
    C = RF.context(C)
    out_dir = work("rich", split, "x").parent
    for old in out_dir.glob("part*.parquet"):
        old.unlink()
    ctx_cols = [c for c in RF.CONTEXT_FEATS if c != "pa"]
    for i, ch in enumerate(F.chunks_by_q(C, size=3_000_000)):
        feats = RF.rich(ch.select("qr", "tr", "pa", *(["y"] if "y" in ch.columns else [])), R, T)
        feats = pl.concat([feats, ch.select(ctx_cols)], how="horizontal")
        feats = feats.with_columns(pl.Series("country_id", R["country"][ch["qr"].to_numpy()]))
        feats.write_parquet(out_dir / f"part{i:03d}.parquet")
        print(f"  {split} part {i}: {feats.height} pairs ({time.time() - t0:.0f}s)", flush=True)


def parts(split: str):
    return sorted(work("rich", split, "x").parent.glob("part*.parquet"))


def load(split: str) -> pl.DataFrame:
    return pl.concat([pl.read_parquet(p) for p in parts(split)])


def feature_names(df: pl.DataFrame, drop=()) -> list:
    return [c for c in df.columns if c not in EXCLUDE and c not in drop]


PARAMS = dict(objective="binary", learning_rate=0.08, num_leaves=255, min_data_in_leaf=100,
              feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=2.0,
              max_bin=255, num_threads=N_JOBS, verbose=-1)


def fold_ids(df: pl.DataFrame, k: int) -> np.ndarray:
    return df.select((pl.col("qr").hash(seed=17) % k).cast(pl.Int8))["qr"].to_numpy()


def train_oof(df: pl.DataFrame, feats: list, k: int = 3, rounds: int = 800, tag: str = "main",
              train_mask=None, params=None):
    """K-fold (grouped by S1) out-of-fold predictions; saves the fold models."""
    params = {**PARAMS, **(params or {})}
    folds = fold_ids(df, k)
    X = df.select(feats).to_numpy().astype(np.float32)
    y = df["y"].to_numpy()
    oof = np.zeros(len(y), dtype=np.float32)
    usable = np.ones(len(y), dtype=bool) if train_mask is None else train_mask
    for f in range(k):
        tr = (folds != f) & usable
        va = folds == f
        t0 = time.time()
        dtr = lgb.Dataset(X[tr], y[tr], feature_name=feats, free_raw_data=False)
        m = lgb.train(params, dtr, rounds)
        oof[va] = m.predict(X[va], num_threads=N_JOBS)
        m.save_model(str(work("models", f"stage_b_{tag}_fold{f}.txt")))
        print(f"  fold {f}: trained on {tr.sum()} rows in {time.time() - t0:.0f}s", flush=True)
    work("models", f"stage_b_{tag}_features.json").write_text(json.dumps(feats))
    return oof


def predict(df: pl.DataFrame, tag: str = "main", k: int = 3) -> np.ndarray:
    feats = json.loads(work("models", f"stage_b_{tag}_features.json").read_text())
    X = df.select(feats).to_numpy().astype(np.float32)
    ps = [lgb.Booster(model_file=str(work("models", f"stage_b_{tag}_fold{f}.txt"))).predict(X, num_threads=N_JOBS)
          for f in range(k)]
    return np.mean(ps, axis=0).astype(np.float32)


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else None)
