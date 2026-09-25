"""Offline evaluation on train with out-of-fold stage B scores."""
import polars as pl

from . import decide
from . import features as F
from .metrics import gt_pairs

BETA2 = 0.25


def macro_f05_idx(pred: pl.DataFrame, gt: pl.DataFrame, s1: pl.DataFrame) -> pl.DataFrame:
    """pred/gt: (qr, tr); s1: (qr, country). Returns per-country and overall macro F0.5."""
    tp = pred.join(gt, on=["qr", "tr"]).group_by("qr").agg(pl.len().alias("tp"))
    np_ = pred.group_by("qr").agg(pl.len().alias("np"))
    ng = gt.group_by("qr").agg(pl.len().alias("ng"))
    t = s1.join(tp, on="qr", how="left").join(np_, on="qr", how="left").join(ng, on="qr", how="left").fill_null(0)
    prec = pl.col("tp") / pl.col("np")
    rec = pl.col("tp") / pl.col("ng")
    f = (pl.when((pl.col("ng") == 0) & (pl.col("np") == 0)).then(1.0)
         .when(pl.col("tp") == 0).then(0.0)
         .otherwise((1 + BETA2) * prec * rec / (BETA2 * prec + rec)))
    t = t.with_columns(f.alias("f"))
    per = t.group_by("country").agg(pl.col("f").mean(), pl.len().alias("n")).sort("country")
    return pl.concat([per, pl.DataFrame({"country": ["ALL"], "f": [t["f"].mean()], "n": [t.height]})],
                     how="vertical_relaxed")


def load_truth(R: pl.DataFrame):
    gt = F.to_index(gt_pairs(), R).select("qr", "tr")
    s1 = R.with_row_index("qr").filter(pl.col("src") == 1).select(pl.col("qr").cast(pl.UInt32), "country")
    return gt, s1


def sweep(S: pl.DataFrame, gt, s1, col="p", taus=(0.3, 0.4, 0.5, 0.6, 0.7, 0.8)):
    """Compare decision rules on scored candidates S (qr, tr, col)."""
    rows = []
    S1 = decide.one_per_target(S, col)
    for tau in taus:
        r = macro_f05_idx(decide.select_threshold(S1, tau, col).select("qr", "tr"), gt, s1)
        rows.append(("1to1+tau=%.2f" % tau, r))
    r = macro_f05_idx(decide.select_expected_f(S1, col).select("qr", "tr"), gt, s1)
    rows.append(("1to1+expectedF", r))
    out = None
    for name, r in rows:
        r = r.select("country", pl.col("f").alias(name))
        out = r if out is None else out.join(r, on="country")
    return out
