"""Stage A: learned candidate filter.

Blocking keeps ~50 candidates per Source 1 record. A small LightGBM model over
cheap similarity features scores every blocked pair and we keep, per S1
record, its best-scoring candidates (plus each target's best S1 records). The
surviving pairs are `candidate_pairs.tsv`: exactly what the final matcher sees.

On train the filter is cross-fitted (2 folds over S1 records) so that the kept
training candidates are scored by a model that never saw their labels.
"""
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from . import features as F
from .config import N_JOBS, work
from .metrics import gt_pairs

PARAMS = dict(objective="binary", learning_rate=0.1, num_leaves=127, min_data_in_leaf=200,
              feature_fraction=0.8, bagging_fraction=0.7, bagging_freq=1, lambda_l2=1.0,
              num_threads=N_JOBS, verbose=-1,
              deterministic=True, force_col_wise=True, seed=0)
ROUNDS = 300
KEEP = dict(top_q=16, top_t=2, floor=0.002)


def labels(P: pl.DataFrame, R: pl.DataFrame) -> pl.DataFrame:
    gt = F.to_index(gt_pairs(), R).select("qr", "tr").with_columns(pl.lit(1, pl.Int8).alias("y"))
    return P.join(gt, on=["qr", "tr"], how="left").with_columns(pl.col("y").fill_null(0))


def build(split: str):
    t0 = time.time()
    R = F.load_records(split)
    P = F.to_index(pl.read_parquet(work(f"{split}_blocks.parquet")), R)
    if split == "train":
        P = labels(P, R)
    out_dir = work("cheap", split, "x").parent
    for old in out_dir.glob("part*.parquet"):
        old.unlink()
    for i, ch in enumerate(F.chunks_by_q(P)):
        F.cheap(ch, R).write_parquet(out_dir / f"part{i:03d}.parquet")
    print(f"cheap features {split}: {P.height} pairs in {time.time() - t0:.0f}s", flush=True)


def _parts(split):
    return sorted(work("cheap", split, "x").parent.glob("part*.parquet"))


def fold_of(qr: pl.Expr, k: int = 2) -> pl.Expr:
    return (qr.hash(seed=11) % k).cast(pl.Int8)


def train():
    """Cross-fitted stage A on train; stores OOF scores and fold models."""
    cols = F.CHEAP_FEATS + ["qr", "tr", "y"]
    # all positives and 10% of negatives: the filter only has to rank candidates
    neg_rate = 0.10
    data = pl.concat([pl.read_parquet(p, columns=cols)
                      .filter((pl.col("y") == 1) | (pl.col("tr").hash(seed=3) % 1000 < neg_rate * 1000))
                      for p in _parts("train")])
    data = data.with_columns(fold_of(pl.col("qr")).alias("fold"))
    models = []
    for k in (0, 1):
        tr = data.filter(pl.col("fold") != k)
        ds = lgb.Dataset(tr.select(F.CHEAP_FEATS).to_numpy(), tr["y"].to_numpy(),
                         feature_name=F.CHEAP_FEATS, free_raw_data=True)
        m = lgb.train(PARAMS, ds, ROUNDS)
        m.save_model(str(work("models", f"stage_a_fold{k}.txt")))
        models.append(m)
        print(f"stage A fold {k} trained on {tr.height} rows", flush=True)
    return models


def score(split: str) -> pl.DataFrame:
    """Score all blocked pairs of a split -> (qr, tr, pa[, y])."""
    models = [lgb.Booster(model_file=str(work("models", f"stage_a_fold{k}.txt"))) for k in (0, 1)]
    out = []
    for p in _parts(split):
        d = pl.read_parquet(p)
        X = d.select(F.CHEAP_FEATS).to_numpy()
        if split == "train":
            fold = d.select(fold_of(pl.col("qr")))["qr"].to_numpy()
            pa = np.where(fold == 0, models[0].predict(X), models[1].predict(X))
        else:
            pa = (models[0].predict(X) + models[1].predict(X)) / 2
        keep_cols = ["qr", "tr"] + (["y"] if "y" in d.columns else [])
        out.append(d.select(keep_cols).with_columns(pl.Series("pa", pa.astype(np.float32))))
    return pl.concat(out)


def prune(S: pl.DataFrame, top_q=KEEP["top_q"], top_t=KEEP["top_t"], floor=KEEP["floor"]):
    rq = pl.col("pa").rank("ordinal", descending=True).over("qr")
    rt = pl.col("pa").rank("ordinal", descending=True).over("tr")
    return S.filter(((rq <= top_q) | (rt <= top_t)) & (pl.col("pa") >= floor))


def run(split: str):
    S = score(split)
    S.write_parquet(work(f"{split}_stage_a_scores.parquet"))
    C = prune(S)
    C.write_parquet(work(f"{split}_cands.parquet"))
    msg = f"{split}: kept {C.height}/{S.height} pairs"
    if "y" in S.columns:
        msg += f", true pairs kept {C['y'].sum()}/{S['y'].sum()}"
    print(msg, flush=True)
    return C


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "build":
        build(sys.argv[2])
    elif cmd == "train":
        train()
    elif cmd == "run":
        run(sys.argv[2])
