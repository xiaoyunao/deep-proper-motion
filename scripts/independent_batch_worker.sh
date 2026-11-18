#!/bin/bash
set -euo pipefail
: "${SLURM_JOB_ID:?Run inside Slurm}"
: "${DEEP_PM_PYTHON:?Set verified Python}"
state=$1
catalogues=$2
sas=$3
mode=$4
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg
rank=${SLURM_PROCID:-0}
ranks=${SLURM_NTASKS:-1}
worker() {
    IFS=$'\t' read -r core box <<< "$1"
    if [[ "$mode" == fetch && -L "$state/inputs/$core" ]] || [[ "$mode" == references && -L "$state/refs/$core" ]]; then
        # prepare verified original receipts, geometry, and all acquisition options.
        return 0
    fi
    if [[ "$mode" == fetch ]]; then
        "$DEEP_PM_PYTHON" -um deep_pm.exposure_inputs "$state/inputs" "$core" "$catalogues" "$sas" "$state/batch-$box.toml" --workers 4
    elif [[ "$mode" == references ]]; then
        "$DEEP_PM_PYTHON" -um deep_pm.continuous_reference extract "$state/refs/$core" "$state/batch-$box.toml" "$core" "$state/inputs/$core"
    else
        return 2
    fi
}
export -f worker
export state catalogues sas mode
awk -v rank="$rank" -v ranks="$ranks" '(NR-1)%ranks==rank' "$state/tasks.tsv" | xargs -r -d '\n' -P2 -I{} bash -c 'worker "$1"' _ '{}'
