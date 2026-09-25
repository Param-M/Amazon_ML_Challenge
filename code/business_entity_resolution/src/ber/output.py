"""Write the two submission files.

matching_results.tsv : source1_entity_id <TAB> matched_entity_ids
candidate_pairs.tsv  : source1_entity_id <TAB> candidate_entity_ids
One row per Source 1 record of the split (in file order), ids comma-separated,
empty when there is nothing to report.
"""
import polars as pl

from .config import OUTPUT_DIR


def _write(ids: pl.DataFrame, pairs: pl.DataFrame, header: tuple, path):
    """ids: (qr, s1) for every S1 record; pairs: (qr, other)."""
    lists = pairs.unique(["qr", "other"]).sort("qr", "other").group_by("qr", maintain_order=True).agg(
        pl.col("other").str.join(",").alias("lst"))
    out = ids.join(lists, on="qr", how="left").sort("qr").select(
        pl.col("s1").alias(header[0]), pl.col("lst").fill_null("").alias(header[1]))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\t".join(header) + "\n")
        for a, b in out.iter_rows():
            f.write(f"{a}\t{b}\n")
    return out.height


def write(R: pl.DataFrame, cands: pl.DataFrame, matches: pl.DataFrame, out_dir=OUTPUT_DIR):
    """R: records (row index = r); cands/matches: (qr, tr) index pairs."""
    rid = R.with_row_index("r").select(pl.col("r").cast(pl.UInt32), "id", "src")
    ids = rid.filter(pl.col("src") == 1).select(pl.col("r").alias("qr"), pl.col("id").alias("s1"))
    to_ids = lambda P: P.join(rid.select(pl.col("r").alias("tr"), pl.col("id").alias("other")), on="tr")
    n1 = _write(ids, to_ids(cands), ("source1_entity_id", "candidate_entity_ids"),
                out_dir / "candidate_pairs.tsv")
    n2 = _write(ids, to_ids(matches), ("source1_entity_id", "matched_entity_ids"),
                out_dir / "matching_results.tsv")
    print(f"wrote {n2} rows -> {out_dir / 'matching_results.tsv'}; {n1} rows -> candidate_pairs.tsv")
