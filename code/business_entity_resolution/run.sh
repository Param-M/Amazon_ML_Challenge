#!/usr/bin/env bash
# Run a pipeline module: ./run.sh ber.<module> [args...]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$HERE/src${PYTHONPATH:+:$PYTHONPATH}"
if [[ "$(uname)" == "Darwin" ]]; then
  # LightGBM needs OpenMP; reuse the copy bundled in scikit-learn's wheel
  # (alternatively: brew install libomp).
  OMP_DIR="$(python -c 'import sklearn, os; print(os.path.join(os.path.dirname(sklearn.__file__), ".dylibs"))')"
  export DYLD_LIBRARY_PATH="$OMP_DIR${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
fi
exec python -m "$@"
