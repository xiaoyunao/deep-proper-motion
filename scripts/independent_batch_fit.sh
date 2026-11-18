#!/bin/bash
set -euo pipefail
: "${SLURM_JOB_ID:?Run inside Slurm}"
: "${DEEP_PM_PYTHON:?Set verified Python}"
state=$1
legacy=$2
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg
rank=${SLURM_PROCID:-0}
ranks=${SLURM_NTASKS:-1}
streams=${DEEP_PM_FIT_STREAMS:-2}
[[ "$streams" =~ ^[1-9][0-9]*$ ]]
worker() {
    IFS=$'\t' read -r core box <<< "$1"
    "$DEEP_PM_PYTHON" -um deep_pm.continuous_fit "$state/outputs/$core" "$core" "$state/inputs/$core" "$state/shared/$box" "$legacy/xiao_$box.csv" "$state/batch-$box.toml" --annual
}
export -f worker
export state legacy
awk -v rank="$rank" -v ranks="$ranks" '(NR-1)%ranks==rank' "$state/tasks.tsv" | xargs -r -d '\n' -P"$streams" -I{} bash -c 'worker "$1"' _ '{}'
