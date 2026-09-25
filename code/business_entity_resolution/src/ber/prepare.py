"""Normalise every record of a split and store one parquet file per split.

Output columns (one row per record, all sources stacked):
  id, src (1/2/3), country,
  n_full, n_core, n_alias, n_legal, n_dom            (name fields)
  a_full, a_street, a_loc, a_unit, a_nums, a_hnum, a_state  (address fields)
"""
import json
import math
import sys
from collections import Counter
from multiprocessing import Pool

import polars as pl

from .config import N_JOBS, raw_path, work
from .text import norm_addr, norm_name

NAME_COLS = ["n_full", "n_core", "n_alias", "n_legal", "n_dom"]
ADDR_COLS = ["a_full", "a_street", "a_loc", "a_unit", "a_nums", "a_hnum", "a_state"]

_TOKEN_MAP = None


def _init(token_map):
    global _TOKEN_MAP
    _TOKEN_MAP = token_map


def _norm_chunk(rows):
    names, addrs, countries = rows
    out_n = [norm_name(n, _TOKEN_MAP) for n in names]
    out_a = [norm_addr(a, c) for a, c in zip(addrs, countries)]
    return out_n, out_a


def load_raw(split: str) -> pl.DataFrame:
    frames = []
    for src in (1, 2, 3):
        cache = work("raw", f"{split}_s{src}.parquet")
        if cache.exists():
            df = pl.read_parquet(cache)
        else:
            df = pl.read_csv(raw_path(split, src), separator="\t", quote_char=None,
                             infer_schema=False, missing_utf8_is_empty_string=True)
            df.write_parquet(cache)
        frames.append(df.with_columns(pl.lit(src, pl.Int8).alias("src")))
    return pl.concat(frames).rename({"entity_id": "id"}).with_columns(
        pl.col("business_name").fill_null(""), pl.col("business_address").fill_null(""),
        pl.col("country").fill_null("").str.strip_chars())


# ------------------------------------------------------------------ domain segmentation

class Segmenter:
    """Unigram Viterbi word segmentation for domain labels ('jayproducts' -> 'jay products')."""

    def __init__(self, counts: Counter, max_len: int = 20):
        total = sum(counts.values())
        self.cost = {w: math.log(total / c) for w, c in counts.items() if len(w) >= 2}
        self.max_len = max_len
        self.unk = math.log(total) + 4.0  # per character cost of unknown text

    def __call__(self, s: str):
        n = len(s)
        best = [0.0] + [math.inf] * n
        back = [0] * (n + 1)
        for i in range(1, n + 1):
            for j in range(max(0, i - self.max_len), i):
                w = s[j:i]
                c = self.cost.get(w)
                if c is None:
                    c = self.unk * (i - j)
                if best[j] + c < best[i]:
                    best[i] = best[j] + c
                    back[i] = j
        words, i = [], n
        while i > 0:
            words.append(s[back[i]:i])
            i = back[i]
        # glue consecutive unknown characters back together
        out = []
        for w in reversed(words):
            if w not in self.cost and out and out[-1] not in self.cost:
                out[-1] += w
            else:
                out.append(w)
        return out


_SEG = None


def _init_seg(seg):
    global _SEG
    _SEG = seg


def _seg_chunk(items):
    out = []
    for full, core in items:
        words = full.split()
        if len(words) == 1 and words[0] not in _SEG.cost:
            seg = _SEG(words[0])
            # accept only if the split produced known words (not one unknown blob)
            if len(seg) > 1 and sum(len(w) for w in seg if w in _SEG.cost) >= 0.6 * len(words[0]):
                s = " ".join(seg)
                out.append((s, s))
                continue
        out.append((full, core))
    return out


def segment_domains(df: pl.DataFrame, min_len: int = 7) -> pl.DataFrame:
    """Split concatenated single-token names ('jayproducts', '#maaconsultants') into words."""
    single = (~pl.col("n_full").str.contains(" ")) & (pl.col("n_full").str.len_chars() >= min_len)
    vocab = Counter()
    for s in df.filter(~single)["n_full"].to_list():
        vocab.update(s.split())
    seg = Segmenter(Counter({w: c for w, c in vocab.items() if c >= 3}))
    sub = df.with_row_index("_r").filter(single).select("_r", "n_full", "n_core")
    items = list(zip(sub["n_full"].to_list(), sub["n_core"].to_list()))
    chunks = [items[i:i + 20000] for i in range(0, len(items), 20000)]
    with Pool(N_JOBS, initializer=_init_seg, initargs=(seg,)) as pool:
        parts = pool.map(_seg_chunk, chunks)
    res = [x for p in parts for x in p]
    upd = pl.DataFrame({"_r": sub["_r"], "seg_full": [r[0] for r in res], "seg_core": [r[1] for r in res]})
    df = df.with_row_index("_r").join(upd, on="_r", how="left")
    return df.with_columns(
        pl.coalesce("seg_full", "n_full").alias("n_full"),
        pl.coalesce("seg_core", "n_core").alias("n_core"),
    ).drop("_r", "seg_full", "seg_core")


def prepare(split: str):
    token_map = json.loads(work("indic_token_map.json").read_text())
    raw = load_raw(split)
    names = raw["business_name"].to_list()
    addrs = raw["business_address"].to_list()
    countries = raw["country"].to_list()
    step = 50000
    chunks = [(names[i:i + step], addrs[i:i + step], countries[i:i + step])
              for i in range(0, len(names), step)]
    with Pool(N_JOBS, initializer=_init, initargs=(token_map,)) as pool:
        results = pool.map(_norm_chunk, chunks, chunksize=1)
    n_rows = [r for res in results for r in res[0]]
    a_rows = [r for res in results for r in res[1]]
    ndf = pl.DataFrame(n_rows, schema=NAME_COLS, orient="row")
    adf = pl.DataFrame(a_rows, schema=ADDR_COLS, orient="row")
    df = pl.concat([raw.select("id", "src", "country"), ndf, adf], how="horizontal")
    df = segment_domains(df)
    out = work(f"{split}_norm.parquet")
    df.write_parquet(out)
    print(f"{split}: {df.height} records -> {out}")
    return df


if __name__ == "__main__":
    for split in sys.argv[1:] or ["train", "test"]:
        prepare(split)
