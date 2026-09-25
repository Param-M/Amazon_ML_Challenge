"""Stage C: agreement features on top of the stage B scores (stacking).

For every Source 1 record, its confident stage B matches form a consensus: the
true copies mostly share one house number, street and name, whereas a group of
look-alike records shares a different (shifted) number. Stage C compares each
candidate with that consensus, and with the competing Source 1 records of the
same target, then re-scores all pairs with LightGBM.

Train uses the out-of-fold stage B scores (each pair scored by a model that
never saw its label); test uses the fold-assigned stage B scores.

Decoy twins. In the test data a look-alike business usually appears as two
records at the same shifted house number (e.g. "... Corp" and "... Westgate"
at 2027 when the real business is at 2026); in train it is mostly a single
record. Two decoys then "support" each other's house number, which in train
only happens for true copies, and a stage C model trained on train alone
accepts them (simulated on train: US 0.988 -> 0.978). Before training we
therefore give a share of train's single off-number negatives a twin: a copy
of the pair with a jittered stage B score. The share per country is measured
without labels by comparing how many single off-number groups the train and
test candidate lists contain (see `twin_rates`).
"""
import sys

import numpy as np
import polars as pl

from . import features as F
from . import stage_b as B
from .config import work

PARAMS = dict(learning_rate=0.05, num_leaves=127, min_data_in_leaf=200)
ROUNDS = 500
TWIN_SEED = 0
TWIN_JITTER = 1.0      # sd of the logit noise between a decoy and its twin

# Upward house-number shifts that are look-alike businesses in >= 85% of the train
# candidate pairs (+1/+2: ~89%, the odd shifts +3..+13 and +21: 97-99%; see
# docs/assets/house_number_shift). Stage C may not raise such a pair above its
# stage B score: on test these are where decoy pairs fooled the consensus.
DECOY_SHIFTS = (1, 2, 3, 4, 5, 7, 9, 11, 13, 21)

CONSENSUS_FEATS = ["p", "c_sum_other", "c_max_other", "c_n_conf_other", "c_rank",
                   "sup_hnum", "sup_street", "sup_name", "is_cons_hnum", "s1_is_cons_hnum",
                   "cons_share", "t_p_max_other", "t_margin", "t_p_rank"]


def _max_other(col: str, group: str) -> pl.Expr:
    first = pl.col(col).max().over(group)
    second = pl.col(col).sort(descending=True).slice(1, 1).first().over(group)
    return pl.when(pl.col(col) < first).then(first).otherwise(second).fill_null(0)


def consensus(S: pl.DataFrame, R: pl.DataFrame) -> pl.DataFrame:
    """S: (qr, tr, p[, src_tr]). Returns (qr, tr, consensus features), same row order.

    `src_tr` (default: tr) is the record whose attributes a row carries; twins
    are extra rows with a fake `tr` and the `src_tr` of the decoy they copy."""
    if "src_tr" not in S.columns:
        S = S.with_columns(pl.col("tr").alias("src_tr"))
    rec = R.select("a_hnum", "a_street", "n_core")
    t = rec[S["src_tr"].to_numpy()]
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
    return d.select("qr", "tr", *CONSENSUS_FEATS).with_columns(
        [pl.col(c).cast(pl.Float32) for c in CONSENSUS_FEATS])


def cap_decoy_shifts(Sc: pl.DataFrame, Sb: pl.DataFrame, R: pl.DataFrame) -> pl.DataFrame:
    """Final scores: stage C, except that pairs at a decoy-style house-number
    shift keep min(stage C, stage B). Sc, Sb: (qr, tr, p) over the same pairs."""
    h = R["a_hnum"]
    x = Sc.select("qr", "tr", "p").join(Sb.select("qr", "tr", pl.col("p").alias("pb")), on=["qr", "tr"],
                                         how="left", maintain_order="left")
    x = x.with_columns(h=h[x["tr"].to_numpy()], h1=h[x["qr"].to_numpy()])
    num = pl.col("h").str.contains(r"^\d+$") & pl.col("h1").str.contains(r"^\d+$")
    shift = pl.when(num).then(pl.col("h").cast(pl.Int64, strict=False) - pl.col("h1").cast(pl.Int64, strict=False))
    capped = shift.is_in(list(DECOY_SHIFTS)) & pl.col("pb").is_not_null()
    return x.select("qr", "tr", pl.when(capped).then(pl.min_horizontal("p", "pb")).otherwise(pl.col("p")).alias("p"))


# ------------------------------------------------------------------ decoy twins

def _single_offnumber(split: str) -> pl.DataFrame:
    """Per country: single off-number groups per S1 record in the candidate lists.
    A group is the candidates of one S1 record sharing one house number that
    differs from the S1 record's own. Label-free."""
    R = F.load_records(split).select("a_hnum", "country", "src")
    C = pl.read_parquet(work(f"{split}_cands.parquet"), columns=["qr", "tr"])
    h = R["a_hnum"]
    d = C.with_columns(h=h[C["tr"].to_numpy()], h1=h[C["qr"].to_numpy()],
                       country=R["country"][C["qr"].to_numpy()])
    d = d.filter((pl.col("h") != "") & (pl.col("h1") != "") & (pl.col("h") != pl.col("h1")))
    g = d.group_by("qr", "h", "country").len().filter(pl.col("len") == 1).group_by("country").len()
    n1 = R.filter(pl.col("src") == 1).group_by("country").len().rename({"len": "n1"})
    return g.join(n1, on="country").select("country", (pl.col("len") / pl.col("n1")).alias("rate"))


def twin_rates() -> dict:
    """Share of train's single off-number groups that test has turned into pairs."""
    a = _single_offnumber("train").rename({"rate": "tr"})
    b = _single_offnumber("test").rename({"rate": "te"})
    r = a.join(b, on="country").with_columns(q=(1 - pl.col("te") / pl.col("tr")).clip(0.0, 0.9))
    return dict(zip(r["country"], r["q"]))


def make_twins(S: pl.DataFrame, R: pl.DataFrame, rates: dict, seed: int = TWIN_SEED) -> pl.DataFrame:
    """S: train (qr, tr, p, y). Returns S with `src_tr` plus twin rows for a share
    of the single off-number negatives (fake tr >= R.height, y = 0)."""
    attrs = R.select("a_hnum", "country")
    t, q = attrs[S["tr"].to_numpy()], attrs[S["qr"].to_numpy()]
    d = S.with_columns(h=t["a_hnum"], h1=q["a_hnum"], c=q["country"])
    off = d.filter((pl.col("h") != "") & (pl.col("h1") != "") & (pl.col("h") != pl.col("h1")))
    g = (off.group_by("qr", "h").agg(n=pl.len(), tr=pl.col("tr").first(), y=pl.col("y").first(),
                                     c=pl.col("c").first())
         .filter((pl.col("n") == 1) & (pl.col("y") == 0)).sort("qr", "h"))
    rng = np.random.default_rng(seed)
    pick = rng.random(g.height) < g["c"].replace_strict(rates, default=0.0).to_numpy()
    tw = g.filter(pl.Series(pick)).select("qr", pl.col("tr").alias("src_tr"))
    tw = tw.with_columns((pl.int_range(pl.len(), dtype=pl.UInt32) + R.height).alias("tr"))
    twin = tw.join(S.rename({"tr": "src_tr"}), on=["qr", "src_tr"], maintain_order="left")
    p = np.clip(twin["p"].to_numpy().astype(np.float64), 1e-6, 1 - 1e-6)
    logit = np.log(p) - np.log1p(-p) + rng.normal(0, TWIN_JITTER, len(p))
    twin = twin.with_columns(pl.Series("p", (1 / (1 + np.exp(-logit))).astype(np.float32)),
                             pl.lit(0).cast(S["y"].dtype).alias("y"))
    cols = ["qr", "tr", "src_tr", "p", "y"]
    return pl.concat([S.with_columns(pl.col("tr").alias("src_tr")).select(cols), twin.select(cols)])


# ------------------------------------------------------------------ features

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


def build_train_twins(S: pl.DataFrame, rates: dict) -> pl.DataFrame:
    """Train frame for stage C: every stage B row plus the decoy twins, with the
    consensus features computed on the augmented candidate lists."""
    R = F.load_records("train")
    Sa = make_twins(S.select("qr", "tr", "p", "y"), R, rates)
    cons = consensus(Sa.select("qr", "tr", "src_tr", "p"), R)
    cons = cons.with_columns(Sa["src_tr"], Sa["y"])
    parts = []
    for p in B.parts("train"):
        d = pl.read_parquet(p).drop("y", "p", strict=False).rename({"tr": "src_tr"})
        parts.append(cons.join(d, on=["qr", "src_tr"], how="inner"))
    return pl.concat(parts).sort("qr", "tr")


def train(rates: dict | None = None):
    """Train stage C (3 folds) on the stage B OOF scores plus decoy twins."""
    rates = twin_rates() if rates is None else rates
    print("  decoy twin rates:", {k: round(v, 3) for k, v in rates.items()}, flush=True)
    df = build_train_twins(pl.read_parquet(work("oof_main.parquet")), rates)
    print(f"  stage C train frame: {df.height} rows ({(df['tr'] != df['src_tr']).sum()} twins)", flush=True)
    feats = B.feature_names(df, drop=("src_tr",))
    p = B.train_oof(df, feats, k=3, rounds=ROUNDS, tag="main", prefix="stage_c", params=PARAMS)
    S = df.select("qr", "tr", "src_tr", "y").with_columns(pl.Series("p", p))
    S.write_parquet(work("oof_c_twins.parquet"))
    # the original candidate lists (no twins) for the in-domain report
    build("train", pl.read_parquet(work("oof_main.parquet")))
    orig = load("train")
    S0 = orig.select("qr", "tr", "y").with_columns(pl.Series("p", _oof_predict(orig, feats)))
    R = F.load_records("train")
    S0 = S0.select("qr", "tr", "y").join(
        cap_decoy_shifts(S0, pl.read_parquet(work("oof_main.parquet")), R), on=["qr", "tr"], maintain_order="left")
    S0.write_parquet(work("oof_c.parquet"))
    return S0, S.select("qr", "tr", "y", "p")


def _oof_predict(df: pl.DataFrame, feats: list, k: int = 3) -> np.ndarray:
    import lightgbm as lgb
    from .config import N_JOBS
    X = df.select(feats).to_numpy().astype(np.float32)
    folds = B.fold_ids(df, k)
    out = np.empty(len(folds), dtype=np.float32)
    for f in range(k):
        m = lgb.Booster(model_file=str(work("models", f"stage_c_main_fold{f}.txt")))
        idx = folds == f
        out[idx] = m.predict(X[idx], num_threads=N_JOBS)
    return out


if __name__ == "__main__":
    train()
