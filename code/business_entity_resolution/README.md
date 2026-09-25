# Business Entity Resolution — Amazon ML Challenge 2026

Blocking + two-stage LightGBM pipeline that links every Source 1 business to
its matching Source 2 / Source 3 records. It regenerates both submission files
from the challenge data:

- `output/matching_results.tsv` — final matches (scored file)
- `output/candidate_pairs.tsv` — the exact candidate set the final model scores

Only the provided training/test files are used. No external data, APIs or
lookups. The only models are LightGBM (MIT license): 2 filter + 3 matcher models,
~1.4M tree nodes, 73 MB in total.

## Setup

Python 3.12, then:

```bash
pip install -r requirements.txt
```

On macOS, LightGBM needs OpenMP. `run.sh` points it at the copy bundled inside
scikit-learn's wheel; alternatively run `brew install libomp`.

## Run end to end

```bash
export BER_DATA_DIR=/path/to/student_resource/dataset   # holds train/ and test/
export BER_WORK_DIR=/path/to/work                       # intermediate files (~20 GB)
export BER_OUTPUT_DIR=/path/to/output                   # the two TSVs
./run.sh ber.pipeline            # all steps
./run.sh ber.pipeline stage_b    # or resume from a step
```

By default the three directories are `data/`, `work/` and `output/` at the
repository root (next to the `code/` folder). `BER_JOBS` sets the number of worker processes and
threads (default: all cores).

A clean run from raw data took about 2 hours (124 min) on a 10-core Apple M5 with
24 GB RAM (peak memory around 15 GB). All LightGBM models are deterministic with fixed
seeds and blocking breaks ties with a fixed hash, so re-runs are reproducible.

## Steps

| step | module | what it does | output (in `WORK_DIR`) |
|---|---|---|---|
| raw | `prepare.load_raw`, `metrics.gt_pairs` | TSV → parquet; ground truth → (s1, other) pairs | `raw/*.parquet` |
| indic | `indic_dict` | learns native-script → Latin word map from aligned training pairs | `indic_token_map.json` |
| prepare | `prepare`, `text`, `geo` | name / address normalisation, domain-name segmentation | `{split}_norm.parquet` |
| block | `blocking` | rare-key blocking via sparse top-k matrix products, within country | `{split}_blocks.parquet` |
| stage_a | `features.cheap`, `stage_a` | cheap similarity features + 2-fold cross-fitted LightGBM filter | `{split}_cands.parquet` |
| stats | `stats` | unsupervised token / modifier statistics per split | `stats/{split}/*` |
| stage_b | `rich`, `stage_b` | rich pair features + competition context | `rich/{split}/part*.parquet` |
| train | `experiment.oof`, `stage_b.train_oof` | 3-fold LightGBM matcher grouped by S1 + OOF evaluation | `models/stage_b_*`, `oof_main.parquet` |
| stage_c | `stage_c` | agreement features from the OOF matcher scores + 3-fold LightGBM re-scorer + OOF evaluation | `models/stage_c_*`, `oof_c.parquet` |
| predict | `predict`, `decide`, `output` | fold-assigned matcher scores → re-scorer; cross-fitted self-training for countries absent from train (France); one S1 per target, expected-F0.5 selection | `output/*.tsv` |

## Source layout

```
src/ber/
  config.py      paths / env vars
  geo.py         state / region name tables (US, India, France) for canonicalisation
  text.py        name + address normalisation
  indic_dict.py  learned native-script word dictionary
  prepare.py     normalises all records, segments concatenated names
  blocking.py    candidate generation (key families, IDF-weighted sparse top-k)
  features.py    cheap pair features, record loading helpers
  stage_a.py     learned candidate filter -> candidate_pairs
  stats.py       unsupervised lookup tables (IDF, name ambiguity, modifier stats)
  rich.py        rich pair features + competition context
  stage_b.py     matcher training helpers
  stage_c.py     agreement features + re-scorer (stacked on the matcher's OOF scores)
  decide.py      one-S1-per-target + expected-F0.5 subset selection
  metrics.py     ground truth + macro F0.5
  evaluate.py    offline evaluation / decision-rule sweep
  experiment.py  OOF and leave-one-country-out experiments
  predict.py     scores test (matcher -> re-scorer, self-training for unseen countries), writes submission
  output.py      TSV writers
  pipeline.py    end-to-end driver
  figures.py     README figures (optional, needs matplotlib; run after the pipeline)
```

Build the submission zip (from the repository root, after the pipeline):

```bash
./make_submission.sh <team_name>   # -> <team_name>_submission.zip
```

Validate the output with the organisers' script:

```bash
python3 utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir dataset/test
```
