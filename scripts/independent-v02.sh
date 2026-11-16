#!/usr/bin/env bash
set -euo pipefail
work=${DEEP_PM_WORK:?set a v0.2 independent Scratch run directory}
python=${DEEP_PM_PYTHON:-python}
export PYTHONPATH="$PWD/src:${DEEP_PM_EXTRA_PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
config=${DEEP_PM_CONFIG:-configs/experiment-a-v02.toml}
for region in "$@"; do
    for route in sdss object gaia; do
        "$python" -u -m deep_pm --work "$work" --config "$config" --region "$region" --route "$route" fetch
    done
    for stage in normalize inventory associate fit validate report; do
        "$python" -u -m deep_pm --work "$work" --config "$config" --region "$region" "$stage"
    done
done
