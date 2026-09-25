"""Stage C: agreement features on top of the stage B scores (stacking).

For every Source 1 record, its confident stage B matches form a consensus: the
true copies mostly share one house number, street and name, whereas a group of
look-alike records shares a different (shifted) number. Stage C compares each
candidate with that consensus, and with the competing Source 1 records of the
same target, then re-scores all pairs with LightGBM.

Train uses the out-of-fold stage B scores (each pair scored by a model that
never saw its label); test uses the fold-averaged stage B scores.
"""
import sys

import polars as pl

from . import features as F
from . import stage_b as B
from .config import work

PARAMS = dict(learning_rate=0.05, num_leaves=127, min_data_in_leaf=200)
ROUNDS = 500

CONSENSUS_FEATS = ["p", "c_sum_other", "c_max_other", "c_n_conf_other", "c_rank",
                   "sup_hnum", "sup_street", "sup_name", "is_cons_hnum", "s1_is_cons_hnum",
                   "cons_share", "t_p_max_other", "t_margin", "t_p_rank"]


def _max_other(col: str, group: str) -> pl.Expr:
    first = pl.col(col).max().over(group)
    second = pl.col(col).sort(descending=True).slice(1, 1).first().over(group)
    return pl.when(pl.col(col) < first).then(first).otherwise(second).fill_null(0)


def consensus(S: pl.DataFrame, R: pl.DataFrame) -> pl.DataFrame:
    """S: (qr, tr, p). Returns S with the consensus features, same row order."""
    rec = R.select("a_hnum", "a_street", "n_core")
    t = rec[S["tr"].to_numpy()]
    h1 = rec["a_hnum"][S["qr"].to_numpy()]
    d = S.with_columns(h=t["a_hnum"], st=t["a_street"], nm=t["n_core"], h1=h1)
    d = d.with_columns(
        (pl.col("p").sum().over("qr") - pl.col("p")).alias("c_sum_other"),
        _max_other("p", "qr").alias("c_max_other"),
        ((pl.col("p") >= 0.5).sum().over("qr") - (pl.col("p") >= 0.5).cast(pl.UInt32)).alias("c_n_conf_other"),
        pl.col("p").rank("ordinal", descending=True).over("qr").alias("c_rank"),
        pl.when(pl.col("h") != "").then(pl.col("p").sum().over(["qr", "h"]) - pl.col("p")).alias("sup_hnum"),
        pl.when(pl.col("st") != "").then(pl.col("p").sum().over(["qr", "st"]) - pl.col("p")).alias("sup_street"),
        (pl.col("p").sum().over(["qr", "nm"]) - pl.col("p")).alias("sup_name"),
        _max_other("p", "tr").alias("t_p_max_other"),
        pl.col("p").rank("ordinal", descending=True).over("tr").alias("t_p_rank"),
    )
    # consensus house number: the number carrying the most stage B probability mass
    mass = (d.filter(pl.col("h") != "").group_by("qr", "h").agg(pl.col("p").sum().alias("m"))
            .sort(["qr", "m", "h"], descending=[False, True, False])
            .group_by("qr", maintain_order=True).first()
            .select("qr", pl.col("h").alias("h_cons"), pl.col("m").alias("m_cons")))
    d = d.join(mass, on="qr", how="left", maintain_order="left")
    d = d.with_columns(
        pl.when(pl.col("h") != "").then((pl.col("h") == pl.col("h_cons")).cast(pl.Float32)).alias("is_cons_hnum"),
        pl.when(pl.col("h1") != "").then((pl.col("h1") == pl.col("h_cons")).cast(pl.Float32)).alias("s1_is_cons_hnum"),
        (pl.col("m_cons") / pl.col("p").sum().over("qr")).alias("cons_share"),
        (pl.col("p") - pl.col("t_p_max_other")).alias("t_margin"),
    )
    return d.select("qr", "tr", *[c for c in CONSENSUS_FEATS]).with_columns(
        [pl.col(c).cast(pl.Float32) for c in CONSENSUS_FEATS])


def build(split: str, scores: pl.DataFrame, tag: str = "c"):
    """Stage B features + consensus features, written in the same parts as stage B."""
    R = F.load_records(split)
    cons = consensus(scores.select("qr", "tr", "p"), R)
    out_dir = work(f"rich_{tag}", split, "x").parent
    for old in out_dir.glob("part*.parquet"):
        old.unlink()
    for i, p in enumerate(B.parts(split)):
        d = pl.read_parquet(p)
        d = d.join(cons, on=["qr", "tr"], how="left", maintain_order="left")
        d.write_parquet(out_dir / f"part{i:03d}.parquet")


def load(split: str, tag: str = "c") -> pl.DataFrame:
    parts = sorted(work(f"rich_{tag}", split, "x").parent.glob("part*.parquet"))
    return pl.concat([pl.read_parquet(p) for p in parts])


def train():
    """Build train stage C features from the stage B OOF scores and train 3 fold models."""
    build("train", pl.read_parquet(work("oof_main.parquet")))
    df = load("train")
    feats = B.feature_names(df)
    p = B.train_oof(df, feats, k=3, rounds=ROUNDS, tag="main", prefix="stage_c", params=PARAMS)
    S = df.select("qr", "tr", "y").with_columns(pl.Series("p", p))
    S.write_parquet(work("oof_c.parquet"))
    return S


if __name__ == "__main__":
    train()
