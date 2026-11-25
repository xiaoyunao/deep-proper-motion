#!/bin/bash
set -euo pipefail
work=$1
code=$2
wrapper=$3
helper=$4
python=$5
runtime=$6
window=$7
cd "$code"
export PYTHONPATH="$code/src:$runtime" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export DPM_COMMIT
DPM_COMMIT=$(git rev-parse HEAD)
exec "$python" "$wrapper" "$work" "$code" "$helper" "$window" --workers 3
