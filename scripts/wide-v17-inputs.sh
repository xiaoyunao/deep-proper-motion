#!/bin/bash
# Acquire new all-band sky and extract frozen C0 references on allocated nodes.
set -euo pipefail
: "${SLURM_JOB_ID:?Run on allocated CPU nodes}"
: "${DEEP_PM_PYTHON:?Set the verified survey environment}"
state=$1
selection=$2
protocol=$3
catalogues=$4
sas=$5
legacy=$6
nodes=${DEEP_PM_BATCH_NODES:-${SLURM_JOB_NUM_NODES:-1}}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
export DEEP_PM_FETCH_STREAMS=${DEEP_PM_FETCH_STREAMS:-1}
export DEEP_PM_REFERENCE_STREAMS=${DEEP_PM_REFERENCE_STREAMS:-2}
mkdir -p "$state/logs"
"$DEEP_PM_PYTHON" -um deep_pm.independent_batch "$state" "$selection" "$protocol"
"$DEEP_PM_PYTHON" -um deep_pm.legacy_regions "$state/catalogues" "$legacy" "$protocol" \
    >"$state/logs/xiao-$SLURM_JOB_ID.out" 2>"$state/logs/xiao-$SLURM_JOB_ID.err" &
legacy_pid=$!
fetch_rc=0
env -u SLURM_CPUS_PER_TASK srun --overlap --exact -N"$nodes" -n"$nodes" -c12 \
    bash scripts/independent_batch_worker.sh "$state" "$catalogues" "$sas" fetch || fetch_rc=$?
if [[ "$fetch_rc" != 0 ]]; then
    wait "$legacy_pid"
    exit "$fetch_rc"
fi
ref_rc=0
env -u SLURM_CPUS_PER_TASK srun --overlap --exact -N"$nodes" -n"$nodes" -c12 \
    bash scripts/independent_batch_worker.sh "$state" "$catalogues" "$sas" references || ref_rc=$?
wait "$legacy_pid"
exit "$ref_rc"
