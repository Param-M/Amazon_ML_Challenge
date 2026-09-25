# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]  
**Team Members:** [List all team members]  
**Submission Date:** [Date]

---

## 1. Executive Summary

We link Source 1 businesses to their Source 2/3 records with a three-stage pipeline:
1. **Blocking.** IDF-weighted blocking on rare name and address keys, computed as sparse top-k matrix products, finds 98.2% of true pairs at about 50 candidates per entity.
2. **Candidate filter (stage A).** A cross-fitted LightGBM filter over cheap similarities cuts candidates to about 8 per entity while keeping 99.996% of the true pairs that survive blocking.
3. **Final matcher (stage B).** A LightGBM model over 89 features scores the remaining pairs. The final match set for each entity is the subset that maximises its **expected F0.5**, after assigning each Source 2/3 record to at most one Source 1 entity.

Out-of-fold macro F0.5 on the full training set is **0.9869** (a clean re-run from raw data gives 0.98687). The main contributions:
- Features that expose how the dataset's look-alike businesses are constructed: a house number shifted by a few units and one name word added or replaced.
- A label-free "modifier statistic" that learns which added words mark a look-alike, computed on each split separately. This lets the model transfer to France, which never appears in training.

---

## 2. Methodology

### 2.1 Problem Analysis

Findings from EDA on the 2.2M / 5.0M / 5.3M training records (Source 1/2/3):

| finding | consequence |
|---|---|
| Every labelled match has the same `country` as its Source 1 record | block within country only |
| No Source 2/3 id appears in two Source 1 match lists | each target belongs to at most one S1 → one-S1-per-target constraint |
| 5.6% of S1 records are singletons; the others have 1–11 matches (mode 3) | the empty prediction matters |
| 27% of Source 2/3 records match nothing. These are look-alikes of a real S1 business: the **same street with the house number moved by a small amount** (2337→2341, 570→573, 11501→11505) plus a **name modifier** ("+ Harbor", "+ Holdings", "Lee → Player") | house-number relationship and "which word was added" are the key decision features |
| True matches carry different noise: truncated/zero-padded/hyphenated numbers (570→57, 6806→006806, 1603→16-03), letter suffixes (170→170B), `null` tokens, state code ↔ name, city typos | number features must distinguish noise from look-alike shifts |
| Names: legal suffix changes, reordering, OCR digits (`8ody`, `Va1ley`), junk prefixes/suffixes (`>>`, `(ID: 51154)`, phone numbers), web domains (`jayproducts.com`), hashtags, d/b/a aliases, acronyms (`AC`), and **made-up trade names** at the same address (`Rizaectonovi`) | normalisation + domain segmentation + address-only matching path |
| Indian names in Devanagari, Tamil, Telugu, Kannada, Malayalam, Bengali, Gujarati, Gurmukhi, Odia (23% of India S2 names); 16 state names in native script | learned transliteration dictionary |
| Test adds France (15%): its own legal forms (SARL, SAS, EURL…), street abbreviations (R, BD, ALL, AV), region ↔ department swaps (Gironde ↔ Nouvelle-Aquitaine) and its own look-alike modifier words (`groupe`, `holding`, `participations`, `ecole`…) | features must not depend on country-specific vocabularies |

Signed house-number difference among candidates with both numbers present (train): positive shifts of +3, +4, +5, +7, +9, +11, +13 and +21 are 97–99% look-alikes, and +1/+2 are about 89%. True matches drift in both directions.

### 2.2 Solution Strategy

**Approach Type:** Blocking + two-stage classifier (learned filter + final matcher) + expected-utility decision rule  
**Core Innovation:** Label-free "modifier statistics" (see §4) and competition-aware context features. Together they let one LightGBM model separate true noisy copies from look-alike businesses, and carry that over to the unseen country (France).

```
TSV ──► normalise ──► blocking (113.7M pairs train / 97.5M test)
                          │
                          ▼
               stage A: cheap features + LightGBM filter  ──► candidate_pairs.tsv
                          │                                    (16.2M train / 15.5M test)
                          ▼
               stage B: 89 rich features + LightGBM (3 folds)
                          │
                          ▼
      one S1 per target  ──►  per-entity expected-F0.5 subset ──► matching_results.tsv
```

---

## 3. Candidate Generation (Blocking)

**Normalisation** (`text.py`):
- **Names:**
  - Native-script words go through a **dictionary learned from the training labels**: 99.998% of native-script pairs have the same word count as the Latin S1 name, so aligning words by position gives 1,347 word translations. Remaining native text falls back to `anyascii`.
  - Strip junk prefixes and suffixes, ids, phone numbers and appended domains.
  - Split d/b/a aliases; undo OCR digit swaps; canonicalise legal forms (inc/llc/ltd/pvt/sarl/sas/…).
  - Split concatenated or domain names into words with a unigram Viterbi segmenter (`jayproducts` → `jay products`).
- **Addresses:**
  - Split into components, then canonicalise the state/region (US codes ↔ names; Indian names, codes and native-script names; French departments → region).
  - Expand abbreviations with country-aware rules (St/Street/Saint, R/Rue, BD, ALL…) and convert ordinals and ordinal words (Eleventh, 11RD → 11).
  - Drop `null` tokens and separate out unit components.
  - Extract all numbers, including rejoined hyphenated forms (16-03 → 1603), and take the first number of the street part as the house number.

- **Blocking keys used** (three families):
  - *name*: core name tokens, sorted token pairs
  - *address*: street tokens, house-number×street-token, locality tokens, number pairs, number×locality token, whole normalised street
  - *mix*: name token×address number, name token×locality token (rare even when both parts are common: "sun marketing" + "3101")
- **Scoring:** a candidate pair's score is the sum of the IDF weights of the keys it shares, with IDF computed on the target side. Keys with target df > 4000 or S1 df > 1500 are skipped. Each S1 record keeps its top 30 targets over all keys plus its top 10 for each key family. Each target keeps its top 4 S1 records over all keys plus its top 2 per family. The top-k is computed with `sparse_dot_topn` (multithreaded sparse matrix product with top-n pruning), so the billions of key-sharing pairs are never materialised. Full train blocking takes about 6 minutes.
- **Stage A filter (last stage before the model, = `candidate_pairs.tsv`):**
  - **Features:** 29 cheap features (rapidfuzz ratios on names and addresses, Jaccard overlaps, house-number equality and difference, state and legal-form agreement, rank within the S1 group).
  - **Model:** LightGBM trained with 2-fold cross-fitting over S1 records, so no training candidate is scored by a model that saw its label.
  - **Kept pairs:** each S1 record's top 16 and each target's top 2, restricted to scores ≥ 0.002.
- **Candidate pairs generated:** 15,541,167 test (9.0 per S1; France 14.2, India 8.9, US 7.1), a reduction ratio of 0.9999991 against all S1 × (S2+S3) pairs; train: 16,166,780 pairs (7.3 per S1).
- **How you ensured true matches were not lost:**

  | stage | train pairs | per S1 | true-pair recall |
  |---|---|---|---|
  | blocking | 113.7M | 51.5 | 98.16% (India 98.09, US 98.21) |
  | stage A filter | 16.2M | 7.3 | 98.16% (keeps 99.996% of blocked positives) |

  Per-family top-k stops same-name look-alikes from crowding out address-only matches (made-up trade names). Of the remaining US misses, 75% are S2/S3 records with **no address**, and 59% carry a name shared by five or more S1 records. Most of these cannot be attributed by any model.

---

## 4. Matching Model

**Features used** (89, all similarities or statistics; none are one-hot country or vocabulary features):
- **Name features:**
  - Scores: rapidfuzz ratio / token-set / token-sort / Jaro-Winkler on core names; ratio without spaces (domains); full-name token-set score; alias (d/b/a) score; consonant-skeleton ratio (transliteration loses vowels).
  - Overlaps: token Jaccard; IDF sums/maxima of shared, extra and missing words; IDF coverage on each side.
  - Soft matching: extra/missing word counts, typo-tolerant.
  - Other: legal-form agreement; digits in the name; OOV share (candidate words never seen in any Source 1 name, a signal for made-up trade names); counts of S1 and target records that share the exact core name (ambiguity).
- **Address features:**
  - Scores: token-set/ratio on the full address, street and locality; unit similarity; address IDF overlaps; state agreement; empty-address flag.
  - House-number features: equality, **signed difference**, relative difference, digit lengths, Levenshtein distance between the digit strings, prefix/truncation relation, letter-suffix pattern (317 → 317C).
  - Other numbers: overlap of the full number sets and the nearest unmatched number.
- **Other:**
  - **Modifier statistics** (`stats.py`, no labels). For every word *w* in a country, over all blocked pairs whose names differ only by adding, deleting or replacing *w*, we compute the fraction that keep the same house number, smoothed and also expressed relative to the country base rate. Look-alike words ("Holdings", "Southside", "Participations") almost never keep the number, while benign noise words ("Services", "Sri", "SARL") do. On train, this rate correlates 0.88 with label precision across 863 words. Because it is recomputed on the test data itself, it adapts to France's own modifier vocabulary.
  - **Competition context:** the stage A score; the pair's rank among the target's S1 candidates and among the S1's candidates; the best competing score on each side; the number of high-scoring competitors; the sum of stage A scores for the S1 record.
  - Source (2 or 3), domain-name flag, native-script flag.

**Model type:** LightGBM binary classifier (255 leaves, learning rate 0.08, 800 rounds, feature fraction 0.7, bagging 0.8). It is trained on all 16.1M stage A candidates with 3 folds grouped by S1 record. Test scores are the mean of the three fold models. The model is MIT-licensed. Stage A and stage B together have about 1.4M tree nodes (73 MB of model files), far under the 8B-parameter limit.

**Threshold selection method:**
1. **One S1 per target:** each Source 2/3 record keeps only its highest-scoring S1.
2. **Expected-F0.5 subset:** for each S1 record, its candidates' probabilities are treated as independent Bernoulli variables. We choose the top-k (k = 0…n) that maximises the **expected per-entity F0.5**, computed exactly with Poisson-binomial dynamic programming (verified against brute-force enumeration). k = 0 wins when the entity is probably a singleton, which is worth a full 1.0. Since the metric is an average of per-entity F0.5, this optimises the leaderboard objective directly, without a global threshold. Out-of-fold it matches or beats every fixed threshold.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** **0.9869** out-of-fold on all 2,206,821 training S1 entities (India 0.9862, US 0.9873)

| configuration (train, OOF) | macro F0.5 |
|---|---|
| stage A score only, best threshold (0.9) | 0.9682 |
| stage B, one-per-target + threshold 0.5 | 0.9858 |
| stage B, one-per-target + threshold 0.7 | 0.9868 |
| **stage B, one-per-target + expected-F0.5** | **0.9869** |

Cross-country transfer (proxy for the unseen France): a stage B model trained only on US scores 0.9733 on India, and one trained only on India scores 0.9774 on US. The same-country OOF scores are 0.9863 and 0.9873. The submitted model is trained on both countries.

Where the remaining 1.31 points of OOF loss come from:

| error type | entities | loss (pts of macro F0.5) |
|---|---|---|
| some true matches missed | 230,618 | 0.84 |
| has matches, predicted none | 5,134 | 0.23 |
| extra wrong match only | 12,800 | 0.13 |
| singleton given a match | 1,789 of 123,247 | 0.08 |

- **Common false positives (wrong merges):**
  - Look-alikes with the house number shifted by 1–2 and the name unchanged, or with a benign-looking word added.
  - Name-only S2/S3 records whose generic name ("Victory Technologies Limited") belongs to a different S1 record with the same name.
  - Same address with a different made-up name, which is sometimes a trade name of the entity and sometimes a different business.
- **Common false negatives (missed matches):** 266k missed true pairs; 141k never reached the candidate set and 125k were rejected by the model.
  - Of the rejected pairs, 58% are Source 2/3 records **with an empty address**.
  - 43% lost the one-S1-per-target assignment to another S1 record, typically one with the same name.
  - Both groups are largely unidentifiable, and declining to predict is the F0.5-optimal choice.
  - The rest are made-up trade names combined with an altered house number.

---

## 6. Conclusion

Most of the result comes from two things. Recall-oriented blocking that runs as sparse matrix products keeps 98.2% of true pairs within ~8 candidates per entity. Features that describe how look-alikes are built — shifted house numbers, and added words whose "look-alike-ness" is estimated without labels — cleanly separate noisy duplicates from sibling businesses. Optimising expected per-entity F0.5 directly handles singletons and ambiguous name-only records in a principled way. The main lesson is that on this data the address *numbers* and the *identity of the changed word* matter more than string similarity itself.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/` holds all source under `src/ber/`, a `README.md` with setup and run instructions, `requirements.txt` (pinned) and `run.sh`. One command regenerates both output files from the challenge data:

```bash
export BER_DATA_DIR=/path/to/student_resource/dataset
./run.sh ber.pipeline
```

Steps: `raw → indic → prepare → block → stage_a → stats → stage_b → train → predict`. A clean run from raw data took 75 minutes on a 10-core Apple M5 with 24 GB RAM.

| module | role |
|---|---|
| `text.py`, `geo.py`, `indic_dict.py`, `prepare.py` | normalisation, learned transliteration, domain segmentation |
| `blocking.py` | key families + IDF sparse top-k blocking |
| `features.py`, `stage_a.py` | cheap features, cross-fitted filter → `candidate_pairs.tsv` |
| `stats.py`, `rich.py`, `stage_b.py` | unsupervised statistics, rich features, final matcher |
| `decide.py`, `predict.py`, `output.py` | one-per-target, expected-F0.5 selection, TSV writers |
| `metrics.py`, `evaluate.py`, `experiment.py` | macro F0.5, decision-rule sweep, OOF / cross-country experiments |

### B. Additional Results

Top stage B features by gain (fold 0): stage A score; the pair's rank among the target's S1 candidates; the number of high-scoring S1 competitors for the target; the best competing S1 score; relative modifier rate of the added word; street similarity; nearest unmatched number; signed house-number difference; modifier count for replaced words; score gap within the S1 group; S1 name ambiguity.

Blocking recall over development iterations (India, train): 91.7% (single key family, name/address tokens) → 97.5% (+ name×address composite keys) → 98.1% (+ per-family top-k, number-pair and number×locality keys, hyphen-joined numbers, concatenated-name segmentation).

Test submission profile: 5,842,917 matched pairs. Per S1 record: France 3.44 matches with 5.1% empty; India 3.35 matches with 5.7% empty; US 3.37 matches with 5.7% empty (training singleton rate: 5.6%).
