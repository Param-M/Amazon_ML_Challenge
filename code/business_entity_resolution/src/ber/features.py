"""Pair features.

`cheap` features are fully vectorised (rapidfuzz batch scoring + polars list
set operations) and are computed for every blocked pair; they feed the
candidate filter (stage A). `rich` features add token-level soft matching,
house-number analysis and neighbourhood context, and feed the final matcher.
Nothing here is country-specific: features are similarities, so they carry
over to countries unseen in training (France).
"""
import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cpdist

from .config import work

REC_COLS = ["n_full", "n_core", "n_alias", "n_legal", "n_dom", "a_full", "a_street", "a_loc",
            "a_unit", "a_nums", "a_hnum", "a_state", "src", "n_indic"]


def load_records(split: str) -> pl.DataFrame:
    """Normalised records in row order; row number == record index `r`."""
    recs = pl.read_parquet(work(f"{split}_norm.parquet"))
    raw = pl.concat([pl.read_parquet(work("raw", f"{split}_s{s}.parquet"))["business_name"]
                     for s in (1, 2, 3)])
    return recs.with_columns(raw.str.contains(r"[ऀ-෿]").alias("n_indic"))


def id_index(recs: pl.DataFrame) -> pl.DataFrame:
    return recs.select(pl.col("id"), pl.int_range(pl.len(), dtype=pl.UInt32).alias("r"))


def to_index(pairs: pl.DataFrame, recs: pl.DataFrame) -> pl.DataFrame:
    """(s1, other) ids -> (qr, tr) record indices, sorted by qr."""
    idx = id_index(recs)
    return (pairs.join(idx.rename({"id": "s1", "r": "qr"}), on="s1")
            .join(idx.rename({"id": "other", "r": "tr"}), on="other")
            .sort("qr", "tr"))


def _rf(a, b, scorer):
    return cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32)


def _nan_if_empty(x: np.ndarray, *cols: pl.Series) -> np.ndarray:
    mask = np.zeros(len(x), dtype=bool)
    for c in cols:
        mask |= (c == "").to_numpy()
    x = x.astype(np.float32)
    x[mask] = np.nan
    return x


def _jaccard(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    inter = a.list.set_intersection(b).list.len()
    union = a.list.set_union(b).list.len()
    return pl.when(union > 0).then(inter / union).otherwise(None)


def cheap(P: pl.DataFrame, R: pl.DataFrame) -> pl.DataFrame:
    """P: (qr, tr) rows. Returns P with cheap feature columns appended."""
    q = R.select(REC_COLS)[P["qr"].to_numpy()]
    t = R.select(REC_COLS)[P["tr"].to_numpy()]
    nc1, nc2 = q["n_core"], t["n_core"]
    af1, af2 = q["a_full"], t["a_full"]
    L = lambda s: s.to_list()
    f = {
        "n_tsr": _rf(L(nc1), L(nc2), fuzz.token_set_ratio),
        "n_ratio": _rf(L(nc1), L(nc2), fuzz.ratio),
        "n_tsort": _rf(L(nc1), L(nc2), fuzz.token_sort_ratio),
        "n_ns": _rf(L(nc1.str.replace_all(" ", "")), L(nc2.str.replace_all(" ", "")), fuzz.ratio),
        "n_jw": _rf(L(nc1), L(nc2), JaroWinkler.normalized_similarity),
        "n_full_tsr": _rf(L(q["n_full"]), L(t["n_full"]), fuzz.token_set_ratio),
        "n_alias_tsr": _nan_if_empty(_rf(L(nc1), L(t["n_alias"]), fuzz.token_set_ratio), t["n_alias"]),
        "a_tsr": _nan_if_empty(_rf(L(af1), L(af2), fuzz.token_set_ratio), af1, af2),
        "a_ratio": _nan_if_empty(_rf(L(af1), L(af2), fuzz.ratio), af1, af2),
        "s_tsr": _nan_if_empty(_rf(L(q["a_street"]), L(t["a_street"]), fuzz.token_set_ratio),
                               q["a_street"], t["a_street"]),
        "l_tsr": _nan_if_empty(_rf(L(q["a_loc"]), L(t["a_loc"]), fuzz.token_set_ratio),
                               q["a_loc"], t["a_loc"]),
    }
    out = P.with_columns(**{k: pl.Series(v) for k, v in f.items()})
    tok = lambda s: s.str.split(" ").list.filter(pl.element() != "")
    h1 = q["a_hnum"].cast(pl.Int64, strict=False)
    h2 = t["a_hnum"].cast(pl.Int64, strict=False)
    side = pl.DataFrame({
        "nt1": tok(nc1), "nt2": tok(nc2), "at1": tok(af1), "at2": tok(af2),
        "nm1": tok(q["a_nums"]), "nm2": tok(t["a_nums"]),
        "h1": h1, "h2": h2, "s1": q["a_state"], "s2": t["a_state"],
        "lg1": q["n_legal"], "lg2": t["n_legal"],
    })
    extra = side.select(
        _jaccard(pl.col("nt1"), pl.col("nt2")).cast(pl.Float32).alias("n_jac"),
        _jaccard(pl.col("at1"), pl.col("at2")).cast(pl.Float32).alias("a_jac"),
        pl.col("nm1").list.set_intersection(pl.col("nm2")).list.len().cast(pl.Float32).alias("num_inter"),
        pl.col("nm1").list.len().cast(pl.Float32).alias("num_n1"),
        pl.col("nm2").list.len().cast(pl.Float32).alias("num_n2"),
        pl.when(pl.col("h1").is_null() | pl.col("h2").is_null()).then(None)
        .otherwise((pl.col("h1") == pl.col("h2")).cast(pl.Float32)).alias("h_eq"),
        (pl.col("h1") - pl.col("h2")).abs().cast(pl.Float32).alias("h_absdiff"),
        pl.when((pl.col("s1") == "") | (pl.col("s2") == "")).then(None)
        .otherwise((pl.col("s1") == pl.col("s2")).cast(pl.Float32)).alias("st_eq"),
        pl.when((pl.col("lg1") == "") | (pl.col("lg2") == "")).then(None)
        .otherwise((pl.col("lg1") == pl.col("lg2")).cast(pl.Float32)).alias("lg_eq"),
        pl.col("nt1").list.len().cast(pl.Float32).alias("n_len1"),
        pl.col("nt2").list.len().cast(pl.Float32).alias("n_len2"),
        (pl.col("at2").list.len() == 0).cast(pl.Float32).alias("a2_empty"),
    )
    out = pl.concat([out, extra], how="horizontal").with_columns(
        pl.Series("src", t["src"].to_numpy().astype(np.float32)),
        pl.Series("n_dom2", (t["n_dom"] != "").to_numpy().astype(np.float32)),
        pl.Series("n_indic2", t["n_indic"].to_numpy().astype(np.float32)),
    )
    # context within the S1 record's candidate list (chunks hold whole S1 groups)
    comb = pl.col("n_tsr") + pl.col("a_tsr").fill_nan(0).fill_null(0)
    out = out.with_columns(
        comb.rank("ordinal", descending=True).over("qr").cast(pl.Float32).alias("q_rank"),
        (comb.max().over("qr") - comb).cast(pl.Float32).alias("q_gap"),
        pl.len().over("qr").cast(pl.Float32).alias("q_n"),
    )
    return out


CHEAP_FEATS = ["n_tsr", "n_ratio", "n_tsort", "n_ns", "n_jw", "n_full_tsr", "n_alias_tsr", "a_tsr",
               "a_ratio", "s_tsr", "l_tsr", "n_jac", "a_jac", "num_inter", "num_n1", "num_n2",
               "h_eq", "h_absdiff", "st_eq", "lg_eq", "n_len1", "n_len2", "a2_empty", "src",
               "n_dom2", "n_indic2", "q_rank", "q_gap", "q_n"]


def chunks_by_q(P: pl.DataFrame, size: int = 4_000_000):
    """Yield slices of P (sorted by qr) that never split an S1 record's candidates."""
    qr = P["qr"].to_numpy()
    start = 0
    n = len(qr)
    while start < n:
        end = min(start + size, n)
        while end < n and qr[end] == qr[end - 1]:
            end += 1
        yield P.slice(start, end - start)
        start = end
