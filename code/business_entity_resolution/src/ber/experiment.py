"""Offline experiments on train: OOF evaluation and cross-country transfer.

python -m ber.experiment oof   -> 3-fold OOF scores (grouped by S1), decision-rule sweep
python -m ber.experiment loco  -> train on one country, evaluate on the other
                                  (proxy for the unseen France test country)
"""
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from . import evaluate as E
from . import features as F
from . import stage_b as B
from .config import N_JOBS, work


def _truth():
    R = F.load_records("train").select("id", "src", "country")
    return E.load_truth(R)


def report(S: pl.DataFrame, label: str):
    """Print the decision-rule sweep for scored train candidates S (qr, tr, p)."""
    gt, s1 = _truth()
    print(f"{label} (out-of-fold):")
    with pl.Config(tbl_width_chars=250, tbl_cols=20):
        print(E.sweep(S, gt, s1, taus=(0.3, 0.4, 0.5, 0.6, 0.7)))


def oof(rounds: int = 800, tag: str = "main", drop=()):
    df = B.load("train")
    feats = B.feature_names(df, drop)
    print(f"{df.height} pairs, {len(feats)} features", flush=True)
    p = B.train_oof(df, feats, k=3, rounds=rounds, tag=tag)
    S = df.select("qr", "tr", "y").with_columns(pl.Series("p", p))
    S.write_parquet(work(f"oof_{tag}.parquet"))
    report(S, "stage B")
    m = lgb.Booster(model_file=str(work("models", f"stage_b_{tag}_fold0.txt")))
    imp = sorted(zip(m.feature_importance("gain"), feats), reverse=True)
    print("top features by gain:", [f"{n}:{g / imp[0][0]:.3f}" for g, n in imp[:40]])


def loco(rounds: int = 600, drop=()):
    df = B.load("train")
    feats = B.feature_names(df, drop)
    gt, s1 = _truth()
    params = {**B.PARAMS}
    res = []
    for src_c, dst_c in (("US", "India"), ("India", "US")):
        t0 = time.time()
        tr = df.filter(pl.col("country_id") == src_c)
        te = df.filter(pl.col("country_id") == dst_c)
        m = lgb.train(params, lgb.Dataset(tr.select(feats).to_numpy().astype(np.float32),
                                          tr["y"].to_numpy(), feature_name=feats), rounds)
        p = m.predict(te.select(feats).to_numpy().astype(np.float32), num_threads=N_JOBS)
        S = te.select("qr", "tr").with_columns(pl.Series("p", p.astype(np.float32)))
        r = E.sweep(S, gt, s1.filter(pl.col("country") == dst_c), taus=(0.3, 0.5, 0.7))
        print(f"train {src_c} -> test {dst_c} ({time.time() - t0:.0f}s)")
        with pl.Config(tbl_width_chars=250, tbl_cols=20):
            print(r.filter(pl.col("country") == dst_c))


if __name__ == "__main__":
    cmd = sys.argv[1]
    drop = tuple(sys.argv[2].split(",")) if len(sys.argv) > 2 and sys.argv[2] else ()
    if cmd == "oof":
        oof(drop=drop)
    elif cmd == "loco":
        loco(drop=drop)
