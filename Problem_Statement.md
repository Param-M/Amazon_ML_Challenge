# Amazon ML Challenge 2026 — Problem Statement

## Business Entity Resolution Challenge

### 1. Overview

In large-scale commercial platforms, business identity data arrives from multiple independent sources. Each source contributes partial and noisy fragments of information about the same real-world entities.

These fragments share no common identifiers. The challenge is to determine which records from different sources refer to the same real-world business.

> **Goal:** Build an ML solution that, given business records from three independent data sources with noisy and inconsistent fields, determines which records across sources refer to the same real-world business entity.

**Source 1** is the deduplicated reference source. For every Source 1 entity, find all matching records from Source 2 and Source 3.

A Source 1 entity may match:
- zero records,
- one record, or
- multiple records

from Source 2 and/or Source 3.

---

## 2. File Format

All challenge files are **tab-separated values (`.tsv`)**.

Submissions must also be tab-separated.

Tabs are required because business addresses and ID-list columns can contain commas.

```python
import pandas as pd

df = pd.read_csv("dataset/train/train_source1.tsv", sep="\t")
```

> **Important:** Reading a `.tsv` without `sep="\t"` can silently produce a single column containing the entire line.

---

## 3. Data Description

Each source file (`*_source1.tsv`, `*_source2.tsv`, `*_source3.tsv`) contains the following columns:

| Column | Description |
|---|---|
| `entity_id` | Unique identifier. Prefix indicates the source: `S1-`, `S2-`, or `S3-`. |
| `business_name` | Business name; may contain abbreviations, legal suffixes, typos, or transliterations. |
| `business_address` | Business address; may contain partial addresses, formatting variations, missing components, or landmark-based references. |
| `country` | Country label. Training contains `US` and `India`; the test set additionally contains `France`. Treat this as an open set of string labels. |

### Country handling

The test set contains **France**, even though France does not occur in the training data.

Therefore:

- Do **not** hard-code the country set to `{US, India}`.
- Do **not** filter out France.
- Do **not** one-hot encode the pipeline in a way that only supports `{US, India}`.
- Every test entity, including French entities, must appear in the submission.

There is no separate `source` column. The source is identified by the `entity_id` prefix and by the file in which the record appears.

---

## 4. Ground Truth

The training ground-truth file, `train_ground_truth.tsv`, contains:

| Column | Description |
|---|---|
| `source1_entity_id` | The `entity_id` of a Source 1 record. |
| `matched_entity_ids` | Comma-separated matching IDs from Source 2 and/or Source 3. Empty when there are no matches. |

---

## 5. Expected Noise Patterns

### Business-name variations

Expect:

- Abbreviations (`Corp` vs. `Corporation`, `Pvt` vs. `Private`, `Ltd` vs. `Limited`)
- Legal-suffix inconsistencies
- DBA/trade names
- Punctuation differences (`&` vs. `and`)
- Word-order changes
- Typos

### Address variations

Expect:

- Abbreviations (`Rd` vs. `Road`, `St` vs. `Street`)
- Transliteration variants
- Missing components
- Missing PIN codes or states
- Landmark-based references such as `Near SBI ATM`
- Municipal numbering variations
- Component reordering

---

## 6. Dataset Details

### Training Dataset

Business records across all three sources with ground-truth matching labels.

### Test Dataset

Business records across all three sources without matching labels.

### Training Files

```text
dataset/train/
├── train_source1.tsv
├── train_source2.tsv
├── train_source3.tsv
└── train_ground_truth.tsv
```

Descriptions:

1. `train_source1.tsv` — Source 1 training records; deduplicated reference source.
2. `train_source2.tsv` — Source 2 training records.
3. `train_source3.tsv` — Source 3 training records.
4. `train_ground_truth.tsv` — Ground-truth matching labels.

### Test Files

```text
dataset/test/
├── test_source1.tsv
├── test_source2.tsv
└── test_source3.tsv
```

Descriptions:

1. `test_source1.tsv` — Source 1 test records. Generate matches for **every** entity.
2. `test_source2.tsv` — Source 2 test records.
3. `test_source3.tsv` — Source 3 test records.

No ground truth is provided for the test set.

To estimate performance locally, create a validation split from the training data and evaluate it using the **F_0.5** metric.

---

# 7. Output Format

The final submission produces two tab-separated files inside the `output/` folder:

```text
output/
├── matching_results.tsv
└── candidate_pairs.tsv
```

### 7.1 `matching_results.tsv`

This is the **only file scored on the leaderboard** and is uploaded to the Portal during the challenge.

| Column | Description |
|---|---|
| `source1_entity_id` | Source 1 `entity_id`. |
| `matched_entity_ids` | Comma-separated matching IDs from Source 2 and/or Source 3. |

Example:

```text
source1_entity_id	matched_entity_ids
S1-00001	S2-00047,S2-00193,S3-00812
S1-00002	S3-00004
S1-00003	
```

### Required rules

- Every Source 1 test entity must have **exactly one row**.
- Leave `matched_entity_ids` empty when there are no matches.
- Do not include duplicate IDs within an ID list.
- IDs must refer only to Source 2 or Source 3 entities that exist in the test set.

---

## 8. `candidate_pairs.tsv`

This contains the candidate set produced by the **blocking / candidate-generation stage** before the final matching model scores candidates.

It must represent the **last candidate set actually passed to the matching model for inference**.

If multiple blocking/filtering stages exist, this file should contain the candidates after the final such stage.

Every ID appearing in `matching_results.tsv` must also appear in `candidate_pairs.tsv`.

### Important

`candidate_pairs.tsv` is **not scored on the leaderboard**.

It is used to analyze:

- blocking quality,
- recall ceiling,
- reduction ratio, and
- pipeline correctness.

Format:

| Column | Description |
|---|---|
| `source1_entity_id` | Source 1 `entity_id`. |
| `candidate_entity_ids` | Comma-separated candidate IDs from Source 2 and/or Source 3. |

Example:

```text
source1_entity_id	candidate_entity_ids
S1-00001	S2-00047,S2-00193,S3-00812,S3-00999
S1-00002	S3-00004
S1-00003	
```

Rules:

- One row per Source 1 entity.
- Empty candidate list when blocking finds no candidates.
- Only S2/S3 IDs.
- No duplicate IDs.
- Final matches must be a subset of the candidates.

---

# 9. Submission Validation

A helper script is provided:

```text
utils/validate_submission.py
```

Run it from the `student_resource/` directory:

```bash
python3 utils/validate_submission.py     --matching output/matching_results.tsv     --candidate output/candidate_pairs.tsv     --test-dir dataset/test
```

The validator:

- prints `PASS` with exit code `0` when the files satisfy the submission rules;
- prints a numbered list of problems with exit code `1` when issues are found.

It checks formatting and consistency only. It does **not** calculate the competition score.

---

# 10. Final Submission Package

In addition to leaderboard uploads, every team submits one ZIP archive containing the code, outputs, and methodology.

Expected structure:

```text
<team_name>_submission.zip
│
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
│
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       ├── README.md
│       └── requirements.txt
│
└── Documentation_template.md
```

### Package requirements

#### `output/`

Contains:

- `matching_results.tsv`
- `candidate_pairs.tsv`

#### `code/business_entity_resolution/`

Must contain a self-contained, runnable copy of the pipeline.

The `src/` directory should contain all source code.

`README.md` must provide exact end-to-end reproduction instructions.

`requirements.txt` (or an equivalent environment file) must pin the required dependencies/versions.

The pipeline should be able to regenerate both output files from the training/test data using only the contents of this folder.

#### Methodology document

Fill in the provided `Documentation_template.md` and include it in the ZIP.

A PDF export is also acceptable.

There is no page limit; prioritize technical depth and clarity.

---

# 11. Constraints

1. Output must follow the exact format described above.
2. Submissions that fail validation will not be evaluated.
3. `matched_entity_ids` may contain only Source 2 and Source 3 entities.
4. Self-matches to Source 1 are not allowed.
5. Every Source 1 test entity must appear in the submission.
6. Duplicate `source1_entity_id` rows are not allowed.
7. Duplicate entity IDs inside an ID list are not allowed.
8. The final model must use a **MIT or Apache 2.0 licensed model**.
9. The final model may contain **up to 8 billion parameters**.

---

# 12. Evaluation

Submissions are evaluated using the **F_0.5 score**, with:

```text
β = 0.5
```

This is a precision-heavy metric.

It penalizes false merges more strongly than missed matches.

## Formula

```text
F_0.5 = (1.25 × Precision × Recall)
        --------------------------------
        (0.25 × Precision + Recall)
```

The score is calculated as a **macro-average**:

1. Calculate F_0.5 for each Source 1 entity.
2. Average the scores across all Source 1 entities.

### Singletons

A Source 1 entity with no true matches is a **singleton**.

- Correctly predicting an empty list → score `1.0` for that entity.
- Predicting any match when there should be none → score `0.0`.

Therefore, correctly identifying entities with **no matches** is important.

### Why the metric is precision-heavy

In real-world entity resolution, incorrectly merging two different businesses can be more damaging than failing to link two records that actually belong together.

F_0.5 therefore gives greater importance to precision.

---

# 13. Example Evaluation

Suppose the model predicts:

```text
S1-00001 → [S2-00047, S2-00193, S3-00812]
```

Ground truth:

```text
S1-00001 → [S2-00047, S3-00812]
```

Then:

```text
Precision = 2 / 3
Recall    = 2 / 2 = 1.0
```

Therefore:

```text
F_0.5 ≈ 0.714
```

---

# 14. Leaderboard

### Public Leaderboard

During the challenge, rankings are based on a subset of the test set and provide real-time feedback.

### Private Leaderboard

After the challenge ends, the private leaderboard evaluates the remaining portion of the test set.

### Final Rankings

Final rankings are based on the **private leaderboard**.

Predictions are submitted for the **full test set** in both cases; the evaluation split is applied during scoring.

---

# 15. Submission Requirements

## Leaderboard submission

During the challenge:

```text
Upload:
matching_results.tsv
```

The file must:

- be tab-separated;
- use the exact required column names;
- contain every Source 1 test entity.

This file drives the public and private leaderboard scores.

## Final package submission

Submit the complete ZIP package containing:

```text
output/
├── matching_results.tsv
└── candidate_pairs.tsv

code/
└── business_entity_resolution/
    ├── src/
    ├── README.md
    └── requirements.txt

Documentation_template.md
```

## Methodology document

The methodology document must describe:

1. Methodology used
2. Candidate generation / blocking strategy
3. Model architecture
4. Feature engineering
5. Other relevant implementation details

The provided `Documentation_template.md` should be used.

---

# 16. Academic Integrity & Fair Play

## Strictly Prohibited: External Data Lookup

Participants **must not use external databases, APIs, or services to look up business identities or resolve entities**.

This includes, but is not limited to:

- Commercial entity-resolution APIs/services
- Government business-registration databases
- Geocoding APIs for address normalization
- External data augmentation from internet sources

### Enforcement

Submitted approaches, methodologies, and code pipelines will be reviewed and verified.

Evidence of external data lookup can result in **immediate disqualification**.

The challenge is intended to test ML and data-science skills using the provided training data.

---

# 17. Recommended Technical Direction

The problem statement highlights several areas worth exploring.

### Candidate generation / blocking

A strong blocking strategy is important because it determines the upper bound of recall.

### String similarity

Potential features include:

- Jaccard similarity
- Levenshtein distance
- TF-IDF cosine similarity

These can be applied to:

- business names
- business addresses

### Country-specific patterns

Pay attention to differences in address structures and formatting between countries.

### Precision vs. recall

Because F_0.5 is precision-heavy, avoid aggressive matching that produces many false merges.

### Singletons

Do not assume every Source 1 entity has a match.

Correctly predicting no match can receive a full score for that entity.

### Submission validation

Always run the provided validator before submitting.

---

# 18. Core Problem in One Sentence

> **For every Source 1 business, identify all corresponding Source 2 and Source 3 records using only the provided data, while minimizing false matches and producing the required candidate and final-match files.**

---

# 19. High-Level ML Pipeline

A practical interpretation of the challenge is:

```text
                 TRAINING DATA
                       │
                       ▼
              Data Understanding
                       │
                       ▼
                Preprocessing
                       │
                       ▼
          ┌────────────────────────┐
          │ Candidate Generation   │
          │       / Blocking       │
          └───────────┬────────────┘
                      │
                      ▼
             Candidate Pairs
                      │
                      ▼
           Similarity / Features
                      │
                      ▼
              Matching Model
                      │
                      ▼
             Match / No Match
                      │
                      ▼
             matching_results.tsv

Candidate-generation output
              │
              ▼
       candidate_pairs.tsv
```

The key challenge is therefore **not simply classification**. It is an **entity-resolution pipeline** combining candidate generation, similarity/feature engineering, and final matching decisions.
