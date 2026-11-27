#!/usr/bin/env bash
set -euo pipefail
: "${PSCRATCH:?verified private scratch required}"
: "${DPM_CHECKOUT:?set an immutable v2.4 checkout}"
: "${DPM_PYTHON:?set the verified scientific interpreter}"
: "${SLURM_JOB_ID:?submit this worker through Slurm}"
work="$PSCRATCH/deep-proper-motion/v2.4"
mkdir -p "$work"/{logs,tmp,mpl,numba}
printf '%s\n' "$SLURM_JOB_ID" >> "$work/private_allocations.txt"
cd "$DPM_CHECKOUT"
export PYTHONPATH="$PWD/src${DPM_RUNTIME:+:$DPM_RUNTIME}"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export NUMBA_CACHE_DIR="$work/numba" MPLCONFIGDIR="$work/mpl" TMPDIR="$work/tmp"
module=$1
shift
exec srun --mpi=none --resv-ports=0 --network=no_vni -N1 -n1 -c "$SLURM_CPUS_PER_TASK" \
    "$DPM_PYTHON" -um "deep_pm.$module" "$@" > "$work/logs/$module-$SLURM_JOB_ID.log" 2>&1
