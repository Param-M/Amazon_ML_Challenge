"""Turn pair probabilities into the final match sets.

1. One S1 per target: a Source 2/3 record belongs to at most one Source 1
   entity (true in every labelled pair), so each target keeps only its
   best-scoring S1 record.
2. Per S1 record, choose the subset of its candidates that maximises the
   *expected* F0.5 of that entity, treating candidate labels as independent
   Bernoulli(p). Because the metric is a per-entity average, this is exactly
   the quantity the leaderboard averages. Candidates are considered in
   decreasing p, so the choice is "take the top k" with k in 0..n; k = 0 wins
   when the entity is probably a singleton (worth a full 1.0).
"""
import numpy as np
import polars as pl

BETA2 = 0.25


def one_per_target(S: pl.DataFrame, col: str = "p") -> pl.DataFrame:
    """Keep, for every target, only its highest-probability S1 record."""
    return S.filter(pl.col(col).rank("ordinal", descending=True).over("tr") == 1)


def _pb_prefix(P: np.ndarray) -> np.ndarray:
    """P: (G, n) probs. Returns pmf[k] of the sum of the first k columns, shape (n+1, G, n+1)."""
    G, n = P.shape
    out = np.zeros((n + 1, G, n + 1))
    pmf = np.zeros((G, n + 1))
    pmf[:, 0] = 1.0
    out[0] = pmf
    for j in range(n):
        p = P[:, j:j + 1]
        new = pmf * (1 - p)
        new[:, 1:] += pmf[:, :-1] * p
        pmf = new
        out[j + 1] = pmf
    return out


def best_k(P: np.ndarray) -> tuple:
    """P: (G, n) probabilities sorted descending per row. Returns (k*, E[F] at k*)."""
    G, n = P.shape
    pre = _pb_prefix(P)                    # TP distribution for prefix k
    suf = _pb_prefix(P[:, ::-1])           # FN distribution for the remaining n-k
    a = np.arange(n + 1)
    best = np.zeros(G)
    arg = np.zeros(G, dtype=np.int64)
    best[:] = suf[n][:, 0]                 # k = 0: score 1 iff nothing is a match
    for k in range(1, n + 1):
        tp = pre[k]                        # (G, n+1) over a
        fn = suf[n - k]                    # (G, n+1) over b
        # f(a, b) = (1+b2) a / (b2 (a + b) + k)
        f = (1 + BETA2) * a[:, None] / (BETA2 * (a[:, None] + a[None, :]) + k)
        e = np.einsum("ga,gb,ab->g", tp, fn, f)
        better = e > best
        best[better] = e[better]
        arg[better] = k
    return arg, best


def select_expected_f(S: pl.DataFrame, col: str = "p", max_n: int = 12, min_p: float = 0.01) -> pl.DataFrame:
    """Per S1 record choose the top-k of its candidates maximising expected F0.5."""
    S = S.filter(pl.col(col) >= min_p).sort(["qr", col, "tr"], descending=[False, True, False])
    S = S.with_columns(pl.int_range(pl.len()).over("qr").alias("_pos")).filter(pl.col("_pos") < max_n)
    g = S.group_by("qr", maintain_order=True).agg(pl.col(col), pl.len().alias("n"))
    keep = []
    for n, grp in g.group_by("n"):
        n = int(n[0])
        P = np.array(grp[col].to_list(), dtype=np.float64).reshape(-1, n)
        k, _ = best_k(P)
        keep.append(pl.DataFrame({"qr": grp["qr"], "_k": k}))
    K = pl.concat(keep)
    return S.join(K, on="qr").filter(pl.col("_pos") < pl.col("_k")).drop("_pos", "_k")


def select_threshold(S: pl.DataFrame, tau: float, col: str = "p") -> pl.DataFrame:
    return S.filter(pl.col(col) >= tau)
