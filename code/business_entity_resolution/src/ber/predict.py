"""Score the test candidates (stage B -> stage C) and write the submission."""
import json
import sys

import lightgbm as lgb
import numpy as np
import polars as pl

from . import decide, output
from . import features as F
from . import stage_b as B
from . import stage_c as C
from .config import N_JOBS, OUTPUT_DIR, work


def _models(prefix: str, tag: str, k: int):
    feats = json.loads(work("models", f"{prefix}_{tag}_features.json").read_text())
    models = [lgb.Booster(model_file=str(work("models", f"{prefix}_{tag}_fold{f}.txt"))) for f in range(k)]
    return feats, models


def score_b(split: str = "test", tag: str = "main", k: int = 3) -> pl.DataFrame:
    """Stage B scores where each S1 record is scored by the fold model its training
    fold would have used (same hash as training). Stage C therefore sees the same
    kind of single-model scores as the out-of-fold train scores it learned from."""
    feats, models = _models("stage_b", tag, k)
    out = []
    for p in B.parts(split):
        d = pl.read_parquet(p, columns=["qr", "tr"] + feats)
        X = d.select(feats).to_numpy().astype(np.float32)
        folds = B.fold_ids(d, k)
        pr = np.empty(len(folds), dtype=np.float32)
        for f, m in enumerate(models):
            idx = folds == f
            if idx.any():
                pr[idx] = m.predict(X[idx], num_threads=N_JOBS)
        out.append(d.select("qr", "tr").with_columns(pl.Series("p", pr)))
    S = pl.concat(out)
    S.write_parquet(work(f"{split}_scores_b.parquet"))
    return S


def score_c(split: str = "test", tag: str = "main", k: int = 3) -> pl.DataFrame:
    """Stage C scores: mean of the 3 stage C fold models."""
    feats, models = _models("stage_c", tag, k)
    out = []
    for p in sorted(work("rich_c", split, "x").parent.glob("part*.parquet")):
        d = pl.read_parquet(p, columns=["qr", "tr"] + feats)
        X = d.select(feats).to_numpy().astype(np.float32)
        pr = np.mean([m.predict(X, num_threads=N_JOBS) for m in models], axis=0)
        out.append(d.select("qr", "tr").with_columns(pl.Series("p", pr.astype(np.float32))))
    S = pl.concat(out)
    S.write_parquet(work(f"{split}_scores_c.parquet"))
    return S


SELF_TRAIN_CONF = 0.98


def unseen_countries() -> set:
    """Countries present in test but absent from train (France in this challenge)."""
    col = lambda split: set(pl.read_parquet(work(f"{split}_norm.parquet"), columns=["country"])["country"].unique())
    return col("test") - col("train")


def self_train(Sc: pl.DataFrame, Sb: pl.DataFrame, countries: set, k: int = 3, rounds: int = 800,
               conf: float = SELF_TRAIN_CONF) -> pl.DataFrame:
    """Cross-fitted self-training of the stage B matcher for countries never seen in training.

    Pairs of those countries that stage C scores >= conf (or <= 1-conf) become
    pseudo-labelled training rows. Fold f's model is trained on the labelled train
    rows and the pseudo rows of the other folds, and re-scores the pairs of fold f,
    so no pair is scored by a model that saw its own pseudo-label. Simulated on
    train (US only -> India): 0.9517 -> 0.9555 macro F0.5 on the unseen country.
    Only the unseen countries' stage B scores are replaced.
    """
    feats, _ = _models("stage_b", "main", k)
    country = F.load_records("test")["country"]
    new = Sc.with_columns(pl.Series("country", country[Sc["qr"].to_numpy()])).filter(pl.col("country").is_in(list(countries)))
    pseudo = (new.filter((pl.col("p") >= conf) | (pl.col("p") <= 1 - conf))
              .select("qr", "tr", (pl.col("p") >= 0.5).cast(pl.Int8).alias("y")))
    test_rows = pl.concat([pl.read_parquet(p, columns=["qr", "tr"] + feats) for p in B.parts("test")])
    target = test_rows.join(new.select("qr", "tr"), on=["qr", "tr"])
    ps = target.join(pseudo, on=["qr", "tr"])
    train = B.load("train").select(["qr", "y"] + feats)
    print(f"  self-training on {sorted(countries)}: {ps.height} pseudo-labelled of {target.height} pairs", flush=True)
    X_tr, y_tr, f_tr = train.select(feats).to_numpy().astype(np.float32), train["y"].to_numpy(), B.fold_ids(train, k)
    X_ps, y_ps, f_ps = ps.select(feats).to_numpy().astype(np.float32), ps["y"].to_numpy(), B.fold_ids(ps, k)
    X_tg, f_tg = target.select(feats).to_numpy().astype(np.float32), B.fold_ids(target, k)
    p_new = np.empty(target.height, dtype=np.float32)
    for f in range(k):
        X = np.concatenate([X_tr[f_tr != f], X_ps[f_ps != f]])
        y = np.concatenate([y_tr[f_tr != f], y_ps[f_ps != f]])
        m = lgb.train(B.PARAMS, lgb.Dataset(X, y, feature_name=feats), rounds)
        idx = f_tg == f
        p_new[idx] = m.predict(X_tg[idx], num_threads=N_JOBS)
        print(f"  self-training fold {f} done", flush=True)
    upd = target.select("qr", "tr").with_columns(pl.Series("p_new", p_new))
    return (Sb.join(upd, on=["qr", "tr"], how="left", maintain_order="left")
            .with_columns(pl.coalesce("p_new", "p").alias("p")).drop("p_new"))


def run(rule: str = "expected_f", out_dir=OUTPUT_DIR, self_training: bool = True):
    Sb = score_b("test")
    C.build("test", Sb)
    S = score_c("test")
    new = unseen_countries()
    if self_training and new:
        Sb = self_train(S, Sb, new)
        Sb.write_parquet(work("test_scores_b_selftrained.parquet"))
        C.build("test", Sb)
        S = score_c("test")
    R = F.load_records("test").select("id", "src", "country")
    cands = pl.read_parquet(work("test_cands.parquet")).select("qr", "tr")
    S1 = decide.one_per_target(S)
    M = decide.select_expected_f(S1) if rule == "expected_f" else decide.select_threshold(S1, float(rule))
    output.write(R, cands, M.select("qr", "tr"), out_dir=out_dir)
    return S, M


if __name__ == "__main__":
    run(*sys.argv[1:2])
