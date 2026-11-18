#!/bin/bash
set -euo pipefail
: "${SLURM_JOB_ID:?Run inside Slurm}"
: "${DEEP_PM_PYTHON:?Set verified Python}"
state=$1
catalogues=$2
sas=$3
mode=$4
streams=${DEEP_PM_REFERENCE_STREAMS:-2}
if [[ "$mode" == fetch ]]; then streams=${DEEP_PM_FETCH_STREAMS:-2}; fi
[[ "$streams" =~ ^[1-9][0-9]*$ ]]
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
        mkdir -p "$state/logs"
        "$DEEP_PM_PYTHON" -um deep_pm.exposure_inputs "$state/inputs" "$core" "$catalogues" "$sas" "$state/batch-$box.toml" --workers 4 >"$state/logs/fetch-$core.log" 2>&1 || { tail -n 20 "$state/logs/fetch-$core.log" >&2; return 1; }
    elif [[ "$mode" == references ]]; then
        "$DEEP_PM_PYTHON" -um deep_pm.continuous_reference extract "$state/refs/$core" "$state/batch-$box.toml" "$core" "$state/inputs/$core"
    else
        return 2
    fi
    printf '%s completed %s\n' "$mode" "$core"
}
export -f worker
export state catalogues sas mode
awk -v rank="$rank" -v ranks="$ranks" '(NR-1)%ranks==rank' "$state/tasks.tsv" | xargs -r -d '\n' -P"$streams" -I{} bash -c 'worker "$1"' _ '{}'
