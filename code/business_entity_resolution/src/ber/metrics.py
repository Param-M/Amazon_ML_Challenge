"""Ground-truth loading and the challenge metric (macro F0.5 over Source 1 entities)."""
import polars as pl

from .config import raw_path, work


def gt_pairs() -> pl.DataFrame:
    """(s1, other) rows for every labelled training match."""
    cache = work("raw", "train_pairs.parquet")
    if cache.exists():
        return pl.read_parquet(cache)
    gt = pl.read_csv(raw_path("train", 1).with_name("train_ground_truth.tsv"), separator="\t",
                     quote_char=None, infer_schema=False)
    pairs = (gt.rename({"source1_entity_id": "s1", "matched_entity_ids": "other"})
             .with_columns(pl.col("other").fill_null("").str.split(",")).explode("other")
             .filter(pl.col("other").is_not_null() & (pl.col("other") != "")))
    pairs.write_parquet(cache)
    return pairs


def candidate_recall(cands: pl.DataFrame, gt: pl.DataFrame, s1_ids=None) -> dict:
    """Pair recall and share of S1 entities whose whole match set is in the candidates."""
    if s1_ids is not None:
        gt = gt.filter(pl.col("s1").is_in(s1_ids.implode()))
        cands = cands.filter(pl.col("s1").is_in(s1_ids.implode()))
    hit = gt.join(cands.select("s1", "other").with_columns(pl.lit(True).alias("hit")),
                  on=["s1", "other"], how="left").with_columns(pl.col("hit").fill_null(False))
    per = hit.group_by("s1").agg(pl.col("hit").all().alias("all"))
    n_s1 = cands["s1"].n_unique() if s1_ids is None else len(s1_ids)
    return {"pair_recall": hit["hit"].mean(), "entity_full_recall": per["all"].mean(),
            "pairs": cands.height, "pairs_per_s1": cands.height / max(n_s1, 1)}


def macro_f05(pred: pl.DataFrame, gt: pl.DataFrame, s1_ids: pl.Series) -> float:
    """pred/gt: (s1, other) rows. Every id in s1_ids is scored (singletons included)."""
    beta2 = 0.25
    base = pl.DataFrame({"s1": s1_ids})
    p = pred.join(base, on="s1").unique(["s1", "other"])
    g = gt.join(base, on="s1")
    tp = p.join(g, on=["s1", "other"]).group_by("s1").agg(pl.len().alias("tp"))
    np_ = p.group_by("s1").agg(pl.len().alias("np"))
    ng = g.group_by("s1").agg(pl.len().alias("ng"))
    t = (base.join(tp, on="s1", how="left").join(np_, on="s1", how="left")
         .join(ng, on="s1", how="left").fill_null(0))
    prec = pl.when(pl.col("np") > 0).then(pl.col("tp") / pl.col("np")).otherwise(0.0)
    rec = pl.when(pl.col("ng") > 0).then(pl.col("tp") / pl.col("ng")).otherwise(0.0)
    f = ((1 + beta2) * prec * rec / (beta2 * prec + rec)).fill_nan(0.0)
    score = (pl.when((pl.col("ng") == 0) & (pl.col("np") == 0)).then(1.0)
             .when(pl.col("tp") == 0).then(0.0).otherwise(f))
    return t.select(score.alias("f"))["f"].mean()
