#!/bin/bash
set -euo pipefail
: "${SLURM_JOB_ID:?Run inside Slurm}"
: "${DEEP_PM_PYTHON:?Set verified Python}"
state=$1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg
export DEEP_PM_M3_WORKERS=${DEEP_PM_M3_WORKERS:-8}
rank=${SLURM_PROCID:-0}
ranks=${SLURM_NTASKS:-1}
while IFS= read -r box; do
    "$DEEP_PM_PYTHON" -um deep_pm.continuous_reference solve "$state/shared/$box" "$state/batch-$box.toml" "$box" "$state/references-$box.json" --separate-components
done < <(cut -f2 "$state/tasks.tsv" | sort -u | awk -v rank="$rank" -v ranks="$ranks" '(NR-1)%ranks==rank')
