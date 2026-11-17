#!/usr/bin/env bash
set -euo pipefail
mode=$1
shift
: "${SLURM_JOB_ID:?use a Slurm allocation}"
rank=${SLURM_PROCID:-0}
nodes=${DEEP_PM_NODE_TASKS:-1}
workers=${DEEP_PM_WORKERS:-3}
py=${DEEP_PM_PYTHON:-python}
root=${DEEP_PM_OUTPUTS:?set the versioned output root}
cfg=${DEEP_PM_CONFIG:-configs/spatial-v08.toml}
mkdir -p "$root/logs"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
pids=()
for ((lane=0; lane<workers; lane++)); do
    (
        failed=0
        i=0
        for field in "$@"; do
            if ((i % (nodes*workers) == rank*workers+lane)); then
                if ! (
                    case "$mode" in
                        preflight)
                            "$py" -u -m deep_pm.regional_preflight "$root" "$DEEP_PM_QSO" "$DEEP_PM_WINDOW" "$field" --config "$cfg"
                            ;;
                        acquire)
                            "$py" -u -m deep_pm.regional_inputs "$root/$field" "$field" "$DEEP_PM_QSO" "$DEEP_PM_SAS" --config "$cfg" --sdss-workers "${DEEP_PM_SDSS_READERS:-1}"
                            ;;
                        reduce)
                            "$py" -u -m deep_pm.regional_production "$root/$field" "$field" "$DEEP_PM_INPUTS/$field" --config "$cfg" --preflight "${DEEP_PM_PREFLIGHT:?set preflight root}/$field"
                            ;;
                        production)
                            "$py" -u -m deep_pm.regional_inputs "$DEEP_PM_INPUTS/$field" "$field" "$DEEP_PM_QSO" "$DEEP_PM_SAS" --config "$cfg" --sdss-workers "${DEEP_PM_SDSS_READERS:-1}" &&
                                "$py" -u -m deep_pm.regional_production "$root/$field" "$field" "$DEEP_PM_INPUTS/$field" --config "$cfg" --preflight "${DEEP_PM_PREFLIGHT:?set preflight root}/$field"
                            ;;
                        audit)
                            "$py" -u -m deep_pm.spatial_audit "$root/$field" "$field" "$DEEP_PM_PRIOR/$field" --config "$cfg"
                            ;;
                        *) exit 2 ;;
                    esac
                ) >"$root/logs/$field-${SLURM_JOB_ID}.log" 2>&1; then
                    echo "Failed $field" >&2
                    failed=1
                else
                    echo "Completed $field on $(hostname)"
                fi
            fi
            i=$((i+1))
        done
        exit "$failed"
    ) &
    pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
done
exit "$failed"
