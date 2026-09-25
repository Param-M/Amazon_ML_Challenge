"""Rich pair features for the final matcher (stage B).

Computed for the stage-A candidates only. Adds, on top of the cheap features:
  * IDF-weighted name/address overlap, rarity of the extra / missing words
  * name ambiguity: how many S1 / S2+S3 records share the exact core name
  * made-up-name signal: candidate words never seen in any Source 1 name
  * modifier statistics of the extra / missing words (see stats.py)
  * house-number relationship: signed difference, digit edit distance,
    truncation, letter suffix, nearest unmatched number
  * soft token alignment (typo-tolerant extra / missing word counts)
  * competition context from stage A scores: best competing S1 for the same
    target, rank of the pair for the S1 and for the target, etc.
"""
import math
from multiprocessing import Pool

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein
from rapidfuzz.process import cpdist

from . import features as F
from .config import N_JOBS

VOWELS = r"[aeiouy]"


def _skeleton(s: pl.Series) -> pl.Series:
    """Consonant skeleton: robust to transliteration vowel loss ('vijy' ~ 'vijay')."""
    s = (s.str.replace_all(VOWELS, "").str.replace_all("ph", "f").str.replace_all("w", "v")
         .str.replace_all("z", "s"))
    for c in "bcdfghjklmnpqrstvx":
        s = s.str.replace_all(c + "{2,}", c)
    return s


def _token_agg(rows: pl.Series, lists: pl.Series, country: pl.Series, table: pl.DataFrame,
               val: str, aggs: dict, keys=("country", "w")) -> pl.DataFrame:
    """Explode per-row token lists, look each token up in `table`, aggregate per row."""
    d = (pl.DataFrame({"row": rows, "w": lists, "country": country}).explode("w")
         .filter(pl.col("w").is_not_null() & (pl.col("w") != "")))
    d = d.join(table, on=list(keys), how="left")
    out = d.group_by("row").agg(**{k: fn(pl.col(val)) for k, fn in aggs.items()})
    return pl.DataFrame({"row": rows}).join(out, on="row", how="left").sort("row")


# ------------------------------------------------------------------ per-pair python

def _suffix_letter(street: str):
    toks = street.split()
    for i, t in enumerate(toks):
        if t.isdigit():
            if i + 1 < len(toks) and len(toks[i + 1]) == 1 and toks[i + 1].isalpha():
                return toks[i + 1]
            return ""
    return None


def _soft_unmatched(a, b):
    """Words of `a` with no exact or near (typo) match in `b`; and how many were near."""
    unmatched = near = 0
    bs = set(b)
    for w in a:
        if w in bs:
            continue
        best = max((Levenshtein.normalized_similarity(w, x) for x in b), default=0.0)
        if best >= 0.75 or any((x.startswith(w) or w.startswith(x)) and min(len(x), len(w)) >= 3 for x in b):
            near += 1
        else:
            unmatched += 1
    return unmatched, near


def _py_rows(args):
    c1, c2, s1, s2, n1, n2, h1, h2 = args
    out = []
    for i in range(len(c1)):
        a, b = c1[i].split(), c2[i].split()
        un2, near2 = _soft_unmatched(b, a)        # extra words in the candidate
        un1, near1 = _soft_unmatched(a, b)        # S1 words missing from the candidate
        x1, x2 = h1[i], h2[i]
        if x1 and x2:
            lev = Levenshtein.distance(x1, x2)
            pre = float(x1 != x2 and (x1.startswith(x2) or x2.startswith(x1)))
        else:
            lev = pre = math.nan
        f1, f2 = _suffix_letter(s1[i]), _suffix_letter(s2[i])
        if f1 is None or f2 is None:
            suf = math.nan
        elif f1 == f2:
            suf = 0.0 if f1 == "" else 1.0
        elif f1 == "":
            suf = 3.0      # only the candidate has a letter suffix (317 -> 317C)
        elif f2 == "":
            suf = 2.0
        else:
            suf = 4.0
        m1 = {int(x) for x in n1[i].split()} if n1[i] else set()
        m2 = {int(x) for x in n2[i].split()} if n2[i] else set()
        u1, u2 = m1 - m2, m2 - m1
        near_num = min((abs(p - q) for p in u2 for q in u1), default=math.nan) if u1 and u2 else math.nan
        d1 = {w for w in a if w.isdigit()}
        d2 = {w for w in b if w.isdigit()}
        ndig = math.nan if not d1 and not d2 else float(d1 == d2)
        out.append((un2, near2, un1, near1, lev, pre, suf, near_num, ndig))
    return out


PY_COLS = ["x_unm2", "x_near2", "x_unm1", "x_near1", "h_lev", "h_prefix", "h_suffix",
           "num_nearest", "n_digits_eq"]


def py_features(q: pl.DataFrame, t: pl.DataFrame) -> pl.DataFrame:
    cols = [q["n_core"], t["n_core"], q["a_street"], t["a_street"], q["a_nums"], t["a_nums"],
            q["a_hnum"], t["a_hnum"]]
    n = len(cols[0])
    step = max(1, n // (N_JOBS * 4) + 1)
    jobs = [tuple(c.slice(s, step).to_list() for c in cols) for s in range(0, n, step)]
    with Pool(N_JOBS) as pool:
        res = pool.map(_py_rows, jobs)
    rows = [r for part in res for r in part]
    return pl.DataFrame(rows, schema=PY_COLS, orient="row").cast(pl.Float32)


# ------------------------------------------------------------------ main

def rich(P: pl.DataFrame, R: pl.DataFrame, T: dict) -> pl.DataFrame:
    """P: (qr, tr, pa[, y]) sorted by qr, whole S1 groups. Returns features."""
    out = F.cheap(P, R)
    cols = F.REC_COLS + ["country"]
    q = R.select(cols)[P["qr"].to_numpy()]
    t = R.select(cols)[P["tr"].to_numpy()]
    n = P.height
    rows = pl.Series("row", np.arange(n, dtype=np.uint32))
    country = q["country"]

    feats = {}
    feats["n_skel"] = cpdist(_skeleton(q["n_core"]).to_list(), _skeleton(t["n_core"]).to_list(),
                             scorer=fuzz.ratio, workers=-1, dtype=np.float32)
    feats["u_tsr"] = cpdist(q["a_unit"].to_list(), t["a_unit"].to_list(),
                            scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)
    feats["u_tsr"][((q["a_unit"] == "") | (t["a_unit"] == "")).to_numpy()] = np.nan
    feats = {k: pl.Series(k, v) for k, v in feats.items()}

    split = lambda s: s.str.split(" ").list.filter(pl.element() != "")
    nt1, nt2 = split(q["n_core"]), split(t["n_core"])
    at1, at2 = split(q["a_full"]), split(t["a_full"])
    L = pl.DataFrame({"nt1": nt1, "nt2": nt2, "at1": at1, "at2": at2})
    L = L.with_columns(
        n_shared=pl.col("nt1").list.set_intersection(pl.col("nt2")),
        n_extra=pl.col("nt2").list.set_difference(pl.col("nt1")),
        n_missing=pl.col("nt1").list.set_difference(pl.col("nt2")),
        a_shared=pl.col("at1").list.set_intersection(pl.col("at2")),
        a_extra=pl.col("at2").list.set_difference(pl.col("at1")),
        a_missing=pl.col("at1").list.set_difference(pl.col("at2")),
    )
    idf_n, idf_a = T["idf_name"], T["idf_addr"]
    agg_sum_max = {"sum": lambda c: c.sum(), "max": lambda c: c.max()}
    parts = []
    for col, tab, pre in (("n_shared", idf_n, "idf_ns"), ("n_extra", idf_n, "idf_nx"),
                          ("n_missing", idf_n, "idf_nm"), ("a_shared", idf_a, "idf_as"),
                          ("a_extra", idf_a, "idf_ax"), ("a_missing", idf_a, "idf_am")):
        g = _token_agg(rows, L[col], country, tab.rename({"idf": "v"}), "v", agg_sum_max)
        parts.append(g.select(pl.col("sum").fill_null(0).alias(pre + "_sum"),
                              pl.col("max").alias(pre + "_max")))
    idf = pl.concat(parts, how="horizontal")
    idf = idf.with_columns(
        n_cov1=pl.col("idf_ns_sum") / (pl.col("idf_ns_sum") + pl.col("idf_nm_sum")),
        n_cov2=pl.col("idf_ns_sum") / (pl.col("idf_ns_sum") + pl.col("idf_nx_sum")),
        a_cov1=pl.col("idf_as_sum") / (pl.col("idf_as_sum") + pl.col("idf_am_sum")),
        a_cov2=pl.col("idf_as_sum") / (pl.col("idf_as_sum") + pl.col("idf_ax_sum")),
    )

    # word never seen in a Source 1 name of this country -> made-up name / typo
    voc = T["s1_vocab"].rename({"s1cnt": "v"})
    oov = _token_agg(rows, nt2, country, voc, "v",
                     {"oov2": lambda c: c.is_null().mean(), "s1cnt_min2": lambda c: c.fill_null(0).min()})

    # modifier statistics of the extra / missing words
    mods = T["modifiers"]
    mod_parts = []
    for col, kind, pre in (("n_extra", "add", "mod_add"), ("n_missing", "del", "mod_del"),
                           ("n_extra", "rep", "mod_rep")):
        tab = mods.filter(pl.col("kind") == kind).drop("kind")
        g = _token_agg(rows, L[col], country, tab, "rate",
                       {"min": lambda c: c.min(), "mean": lambda c: c.mean()})
        g2 = _token_agg(rows, L[col], country, tab, "rel", {"min": lambda c: c.min()})
        g3 = _token_agg(rows, L[col], country, tab, "logn", {"min": lambda c: c.min()})
        mod_parts.append(pl.DataFrame({pre + "_min": g["min"], pre + "_mean": g["mean"],
                                       pre + "_rel": g2["min"], pre + "_logn": g3["min"]}))
    mod = pl.concat(mod_parts, how="horizontal")

    # ambiguity of the names
    nc = T["name_counts"]
    amb1 = (pl.DataFrame({"country": country, "n_core": q["n_core"]})
            .join(nc, on=["country", "n_core"], how="left")
            .select(pl.col("cnt_s1").alias("s1_same_name1"), pl.col("cnt_t").alias("t_same_name1")))
    amb2 = (pl.DataFrame({"country": country, "n_core": t["n_core"]})
            .join(nc, on=["country", "n_core"], how="left")
            .select(pl.col("cnt_s1").alias("s1_same_name2"), pl.col("cnt_t").alias("t_same_name2")))

    h1 = q["a_hnum"].cast(pl.Int64, strict=False)
    h2 = t["a_hnum"].cast(pl.Int64, strict=False)
    num = pl.DataFrame({"h1": h1, "h2": h2}).select(
        (pl.col("h2") - pl.col("h1")).cast(pl.Float32).alias("h_diff"),
        ((pl.col("h2") - pl.col("h1")).abs() / pl.max_horizontal("h1", "h2").clip(1)).cast(pl.Float32).alias("h_reldiff"),
        pl.col("h1").cast(pl.Utf8).str.len_chars().cast(pl.Float32).alias("h_len1"),
        pl.col("h2").cast(pl.Utf8).str.len_chars().cast(pl.Float32).alias("h_len2"),
    )
    py = py_features(q, t)

    out = pl.concat([out, pl.DataFrame(list(feats.values())), idf, oov.drop("row"), mod,
                     amb1, amb2, num, py], how="horizontal")
    out = out.with_columns(
        (pl.col("num_inter") / pl.col("num_n2")).cast(pl.Float32).alias("num_cov2"),
        (pl.col("num_inter") / pl.col("num_n1")).cast(pl.Float32).alias("num_cov1"),
    )
    return out


CONTEXT_FEATS = ["pa", "q_pa_rank", "q_pa_max_other", "q_pa_sum", "q_n_high", "t_pa_rank",
                 "t_pa_max_other", "t_n", "t_n_high"]


def context(S: pl.DataFrame) -> pl.DataFrame:
    """Competition features from stage-A scores over the kept candidates (qr, tr, pa)."""
    top2 = lambda g: pl.col("pa").top_k(2).over(g)
    S = S.with_columns(
        pl.col("pa").rank("ordinal", descending=True).over("qr").cast(pl.Float32).alias("q_pa_rank"),
        pl.col("pa").sum().over("qr").cast(pl.Float32).alias("q_pa_sum"),
        (pl.col("pa") > 0.5).sum().over("qr").cast(pl.Float32).alias("q_n_high"),
        pl.col("pa").rank("ordinal", descending=True).over("tr").cast(pl.Float32).alias("t_pa_rank"),
        pl.len().over("tr").cast(pl.Float32).alias("t_n"),
        (pl.col("pa") > 0.5).sum().over("tr").cast(pl.Float32).alias("t_n_high"),
    )
    # best score among the *other* candidates of the same S1 / same target
    for g, name in (("qr", "q_pa_max_other"), ("tr", "t_pa_max_other")):
        first = pl.col("pa").max().over(g)
        second = pl.col("pa").sort(descending=True).slice(1, 1).first().over(g)
        S = S.with_columns(pl.when(pl.col("pa") < first).then(first).otherwise(second)
                           .fill_null(0).cast(pl.Float32).alias(name))
    return S
