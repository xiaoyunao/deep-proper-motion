#!/usr/bin/env bash
set -euo pipefail
work=${DEEP_PM_WORK:?set DEEP_PM_WORK to the NERSC Scratch project location}
python=${DEEP_PM_PYTHON:-python}
region=${1:-smoke}
export PYTHONPATH="$PWD/src:${DEEP_PM_EXTRA_PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for stage in normalize inventory associate fit validate report; do
    printf '%s %s\n' "$region" "$stage"
    "$python" -u -m deep_pm --work "$work" --region "$region" "$stage"
done
