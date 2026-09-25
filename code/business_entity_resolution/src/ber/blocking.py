"""Candidate generation (blocking).

Every record is described by a bag of blocking keys built from its name and
address (see `record_keys`). A Source 1 record and a Source 2/3 record become a
candidate pair when they share keys; the pair score is the summed IDF weight of
the shared keys, so rare shared tokens count most. Keys too common to be
informative (document frequency above a cap) are skipped.

Keys come in three families, and top-k lists are taken per family as well as
over all keys, so that e.g. a record that only shares its address (a trade name
that differs completely from the Source 1 name) is not crowded out by dozens of
same-name look-alikes:
  name : name tokens, sorted name-token pairs
  addr : street tokens, house number x street token, locality tokens, number
         pairs, number x locality token, whole street string
  mix  : name token x address number, name token x locality token
Blocking is always within one country (true matches never cross countries).
The top-k selection runs as a sparse matrix product (sparse_dot_topn), which
never materialises the full list of key-sharing pairs.
"""
import math
import sys
import time

import numpy as np
import polars as pl
import scipy.sparse as sp
from sparse_dot_topn import sp_matmul_topn

from .config import N_JOBS, work

GENERIC_ADDR = {"st", "rd", "ave", "dr", "ln", "ct", "blvd", "hwy", "pkwy", "pl", "cir", "trl",
                "ter", "sq", "aly", "way", "rue", "allee", "chemin", "imp", "rte", "nr", "opp",
                "floor", "plot", "door", "h", "hno", "unit", "apt", "flat", "bldg", "n", "s", "e",
                "w", "ne", "nw", "se", "sw", "no", "bis", "main", "cross", "sec", "blk", "col",
                "vill", "near", "po", "box", "pmb", "rm", "dist", "road", "street"}

# per family: (top-k per S1 record, top-k per target record)
DEFAULTS = dict(cap_t=4000, cap_q=1500,
                topk={"all": (30, 4), "name": (10, 2), "addr": (10, 2), "mix": (10, 2)})


def _tok_list(col: str) -> pl.Expr:
    return pl.col(col).str.split(" ")


FAMS = {"name": 0, "addr": 1, "mix": 2}


def _k(df: pl.DataFrame, expr: pl.Expr, fam: str) -> pl.DataFrame:
    return df.select(pl.col("i").cast(pl.UInt32), expr.hash(seed=7).alias("key"),
                     pl.lit(FAMS[fam], pl.UInt8).alias("fam"))


def record_keys(df: pl.DataFrame) -> pl.DataFrame:
    """Return (i, key, fam) rows; `i` is the record's row index inside `df`."""
    name_tok = (df.select("i", pl.concat_list(_tok_list("n_core"), _tok_list("n_alias")).alias("t"))
                .explode("t").filter(pl.col("t").str.len_chars() >= 2).unique())
    street = (df.select("i", "a_hnum", _tok_list("a_street").alias("t")).explode("t")
              .filter((pl.col("t").str.len_chars() >= 2) & ~pl.col("t").str.contains(r"^\d+$")
                      & ~pl.col("t").is_in(list(GENERIC_ADDR))).unique())
    loc = (df.select("i", _tok_list("a_loc").alias("l")).explode("l")
           .filter((pl.col("l").str.len_chars() >= 3) & ~pl.col("l").is_in(list(GENERIC_ADDR)))
           .unique())
    nums = (df.select("i", pl.col("a_nums").str.split(" ").list.head(6).alias("x")).explode("x")
            .filter(pl.col("x").str.len_chars() >= 1).unique())
    keys = [
        _k(name_tok, "n|" + pl.col("t"), "name"),
        _k(name_tok.join(name_tok, on="i", suffix="2").filter(pl.col("t") < pl.col("t2")),
           "p|" + pl.col("t") + "|" + pl.col("t2"), "name"),
        _k(street, "a|" + pl.col("t"), "addr"),
        _k(street.filter(pl.col("a_hnum") != ""), "h|" + pl.col("a_hnum") + "|" + pl.col("t"), "addr"),
        _k(loc, "l|" + pl.col("l"), "addr"),
        _k(nums.join(nums, on="i", suffix="2").filter(pl.col("x").cast(pl.Int64, strict=False)
                                                       < pl.col("x2").cast(pl.Int64, strict=False)),
           "nn|" + pl.col("x") + "|" + pl.col("x2"), "addr"),
        _k(nums.join(loc, on="i"), "z|" + pl.col("x") + "|" + pl.col("l"), "addr"),
        _k(df.filter(pl.col("a_street").str.contains(" ")), "s|" + pl.col("a_street"), "addr"),
        _k(name_tok.join(nums, on="i"), "x|" + pl.col("t") + "|" + pl.col("x"), "mix"),
        _k(name_tok.join(loc, on="i"), "y|" + pl.col("t") + "|" + pl.col("l"), "mix"),
    ]
    return pl.concat(keys).unique(["i", "key"])


def _csr(keys: pl.DataFrame, n_rows: int, n_cols: int, weighted: bool):
    data = keys["w"].to_numpy().astype(np.float64) if weighted else np.ones(keys.height)
    return sp.csr_matrix((data, (keys["i"].to_numpy(), keys["k"].to_numpy())),
                         shape=(n_rows, n_cols), dtype=np.float64)


def _topn(A, B, top_n: int, chunk: int = 20000):
    """Row-wise top-n of A @ B, as (row, col, score) numpy arrays."""
    rows, cols, vals = [], [], []
    for start in range(0, A.shape[0], chunk):
        r = sp_matmul_topn(A[start:start + chunk], B, top_n=top_n, n_threads=N_JOBS).tocoo()
        rows.append(r.row.astype(np.int64) + start)
        cols.append(r.col)
        vals.append(r.data)
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)


TIE_EPS = 1e-6


def _tiebreak(n: int, salt: int = 0) -> np.ndarray:
    """Deterministic pseudo-random value in [0, 1) per record index (salt = key family)."""
    mix = np.uint64((salt * 0x632BE59BD9B4E019) & 0xFFFFFFFFFFFFFFFF)
    idx = np.arange(n, dtype=np.uint64) + mix
    x = (idx * np.uint64(0x9E3779B97F4A7C15)) >> np.uint64(40)
    return x.astype(np.float64) / float(1 << 24)


def _with_tiebreak(M, u: np.ndarray):
    """Scale row i by (1 + eps * u_i): scores become S * (1 + eps u_q)(1 + eps u_t).

    Candidates with equal scores at the top-k cut-off would otherwise be chosen
    by thread scheduling; this makes the candidate set identical from run to run
    without adding any non-zero entries (sparsity is unchanged).
    """
    M = M.tocsr(copy=True)
    M.data *= np.repeat(1 + TIE_EPS * u, np.diff(M.indptr))
    return M


def block_country(q: pl.DataFrame, t: pl.DataFrame, cap_t, cap_q, topk):
    """q: S1 records, t: S2/S3 records of one country. Returns (qi, ti) candidate rows."""
    qk, tk = record_keys(q), record_keys(t)
    n_t = t.height
    dft = tk.group_by("key").agg(pl.len().alias("dft"))
    dfq = qk.group_by("key").agg(pl.len().alias("dfq"))
    keep = (dft.join(dfq, on="key").filter((pl.col("dft") <= cap_t) & (pl.col("dfq") <= cap_q))
            .with_columns((pl.lit(math.log(n_t + 1)) - pl.col("dft").cast(pl.Float64).log())
                          .alias("w"))
            .sort("key").with_row_index("k"))   # fixed column order -> fixed summation order
    qk = qk.join(keep.select("key", "k", "w"), on="key")
    tk = tk.join(keep.select("key", "k", "w"), on="key")
    out = []
    for salt, (fam, (k_fwd, k_rev)) in enumerate(topk.items()):
        # each family breaks ties differently, so tied candidates are spread across families
        uq, ut = _tiebreak(q.height, salt), _tiebreak(t.height, salt + 101)
        qf = qk if fam == "all" else qk.filter(pl.col("fam") == FAMS[fam])
        tf = tk if fam == "all" else tk.filter(pl.col("fam") == FAMS[fam])
        Q = _with_tiebreak(_csr(qf, q.height, keep.height, weighted=True), uq)
        T = _with_tiebreak(_csr(tf, t.height, keep.height, weighted=False), ut)
        fr, fc, _ = _topn(Q, T.T.tocsr(), k_fwd)       # S1 -> best targets
        rr, rc, _ = _topn(T, Q.T.tocsr(), k_rev)       # target -> best S1
        out.append(pl.DataFrame({"qi": np.concatenate([fr, rc]).astype(np.uint32),
                                 "ti": np.concatenate([fc, rr]).astype(np.uint32)}))
    return pl.concat(out).unique()


def run(split: str, **kw):
    cfg = {**DEFAULTS, **kw}
    df = pl.read_parquet(work(f"{split}_norm.parquet"))
    out = []
    for country in df["country"].unique().sort().to_list():
        t0 = time.time()
        d = df.filter(pl.col("country") == country)
        q = d.filter(pl.col("src") == 1).with_row_index("i")
        t = d.filter(pl.col("src") != 1).with_row_index("i")
        if q.height == 0 or t.height == 0:
            continue
        pr = block_country(q, t, **cfg)
        pr = (pr.join(q.select(pl.col("i").alias("qi"), pl.col("id").alias("s1")), on="qi")
              .join(t.select(pl.col("i").alias("ti"), pl.col("id").alias("other")), on="ti")
              .select("s1", "other"))
        print(f"  {split}/{country}: {q.height} S1 x {t.height} targets -> {pr.height} pairs "
              f"({pr.height / q.height:.1f}/S1) in {time.time() - t0:.0f}s", flush=True)
        out.append(pr)
    res = pl.concat(out)
    res.write_parquet(work(f"{split}_blocks.parquet"))
    return res


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "train")
