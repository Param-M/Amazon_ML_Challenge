"""End-to-end pipeline: challenge TSVs -> output/matching_results.tsv + output/candidate_pairs.tsv.

    python -m ber.pipeline            # run every step
    python -m ber.pipeline stage_b    # run from a given step onwards

Steps (each reads the previous step's files from WORK_DIR):
  raw      cache the TSVs as parquet, explode the training labels into pairs
  indic    learn the native-script -> Latin word dictionary (train labels only)
  prepare  normalise names / addresses of every record (train and test)
  block    candidate generation by rare shared keys (~50 candidates / S1)
  stage_a  cheap features + cross-fitted LightGBM filter (~8 candidates / S1)
  stats    unsupervised token / modifier statistics per split
  stage_b  rich pair features for the stage A candidates
  train    3-fold LightGBM matcher (grouped by S1) + out-of-fold evaluation
  predict  score test candidates, one S1 per target, expected-F0.5 selection,
           write both TSVs
"""
import sys
import time

from . import blocking, experiment, indic_dict, metrics, predict, prepare, stage_a, stage_b, stats


def step_raw():
    for split in ("train", "test"):
        prepare.load_raw(split)
    metrics.gt_pairs()


def step_indic():
    indic_dict.main()


def step_prepare():
    for split in ("train", "test"):
        prepare.prepare(split)


def step_block():
    for split in ("train", "test"):
        blocking.run(split)


def step_stage_a():
    for split in ("train", "test"):
        stage_a.build(split)
    stage_a.train()
    for split in ("train", "test"):
        stage_a.run(split)


def step_stats():
    for split in ("train", "test"):
        stats.build(split)


def step_stage_b():
    for split in ("train", "test"):
        stage_b.build(split)


def step_train():
    experiment.oof()


def step_predict():
    predict.run("main")


STEPS = [("raw", step_raw), ("indic", step_indic), ("prepare", step_prepare), ("block", step_block),
         ("stage_a", step_stage_a), ("stats", step_stats), ("stage_b", step_stage_b),
         ("train", step_train), ("predict", step_predict)]


def main(start: str = "raw"):
    names = [n for n, _ in STEPS]
    t_all = time.time()
    for name, fn in STEPS[names.index(start):]:
        t0 = time.time()
        print(f"=== {name}", flush=True)
        fn()
        print(f"=== {name} done in {time.time() - t0:.0f}s", flush=True)
    print(f"pipeline finished in {(time.time() - t_all) / 60:.1f} min")


if __name__ == "__main__":
    main(*sys.argv[1:2])
