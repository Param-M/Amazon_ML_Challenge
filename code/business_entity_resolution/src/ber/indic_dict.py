"""Learn a native-script -> Latin token dictionary from the training labels.

Source 2/3 often write Indian business names in Devanagari, Tamil, Telugu, etc.
In 99.99% of labelled pairs the native-script name has exactly as many words
as the Latin Source 1 name (it is a word-by-word transliteration), so aligning
words by position gives clean (native word, Latin word) pairs. For each native
word we keep the most frequent Latin word. Only training data is used.
"""
import json
import re
from collections import Counter, defaultdict

import polars as pl

from .config import work

TOKEN_RE = re.compile(r"[ऀ-෿‌‍]+|[A-Za-z0-9]+")
INDIC_RE = re.compile(r"[ऀ-෿]")


def learn(min_count: int = 2, min_share: float = 0.5) -> dict:
    s1 = pl.read_parquet(work("raw", "train_s1.parquet")).select(
        pl.col("entity_id").alias("s1"), pl.col("business_name").alias("n1"))
    oth = pl.concat([pl.read_parquet(work("raw", f"train_s{s}.parquet")) for s in (2, 3)])
    oth = oth.filter(pl.col("business_name").str.contains(r"[ऀ-෿]"))
    pairs = pl.read_parquet(work("raw", "train_pairs.parquet"))
    j = oth.join(pairs, left_on="entity_id", right_on="other").join(s1, on="s1")

    counts = defaultdict(Counter)
    for n1, n2 in zip(j["n1"].to_list(), j["business_name"].to_list()):
        a, b = TOKEN_RE.findall(n1), TOKEN_RE.findall(n2)
        if len(a) != len(b):
            continue
        for x, y in zip(a, b):
            if INDIC_RE.search(y) and not INDIC_RE.search(x):
                counts[y][x.lower()] += 1

    mapping = {}
    for native, c in counts.items():
        latin, n = c.most_common(1)[0]
        total = sum(c.values())
        if n >= min_count and n / total >= min_share:
            mapping[native] = latin
    return mapping


def main():
    mapping = learn()
    out = work("indic_token_map.json")
    out.write_text(json.dumps(mapping, ensure_ascii=False))
    print(f"learned {len(mapping)} native-script tokens -> {out}")


if __name__ == "__main__":
    main()
