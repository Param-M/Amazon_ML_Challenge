#!/usr/bin/env bash
# Build <team>_submission.zip in the layout the organisers require:
#   output/{matching_results,candidate_pairs}.tsv
#   code/business_entity_resolution/
#   Documentation_template.md            (= docs/METHODOLOGY.md)
# Usage: ./make_submission.sh <team_name>      (run the pipeline first)
set -euo pipefail
TEAM="${1:?usage: ./make_submission.sh <team_name>}"
ROOT="$(cd "$(dirname "$0")" && pwd)"
for f in output/matching_results.tsv output/candidate_pairs.tsv; do
  [[ -f "$ROOT/$f" ]] || { echo "missing $f - run the pipeline first" >&2; exit 1; }
done
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$STAGE/output" "$STAGE/code"
cp "$ROOT/output/matching_results.tsv" "$ROOT/output/candidate_pairs.tsv" "$STAGE/output/"
rsync -a --exclude '__pycache__' --exclude '*.pyc' "$ROOT/code/business_entity_resolution" "$STAGE/code/"
cp "$ROOT/docs/METHODOLOGY.md" "$STAGE/Documentation_template.md"
rm -f "$ROOT/${TEAM}_submission.zip"
(cd "$STAGE" && zip -qr "$ROOT/${TEAM}_submission.zip" output code Documentation_template.md)
echo "wrote ${TEAM}_submission.zip"
