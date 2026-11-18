#!/bin/bash
# Submit with verified local account/QOS; no data paths are committed here.
set -euo pipefail
: "${SLURM_JOB_ID:?Run inside Slurm}"
: "${DEEP_PM_PYTHON:?Set the verified Python executable}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg
frozen=$1
output=$2
legacy=$3
config=$4
line=$(( ${SLURM_ARRAY_TASK_ID:-0} + 1 ))
IFS=$'\t' read -r core source reference < <(sed -n "${line}p" "$frozen/run-plan.tsv")
[[ -n "$core" && -d "$source" && -d "$reference" ]]
"$DEEP_PM_PYTHON" -m deep_pm.continuous_conditions "$output/$core" "$source" "$frozen/shared" "$legacy" "$config" "$core"
