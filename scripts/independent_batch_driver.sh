#!/bin/bash
# Allocated CPU nodes with explicitly bounded download/reference/fit streams.
set -euo pipefail
: "${SLURM_JOB_ID:?Run inside Slurm}"
: "${DEEP_PM_PYTHON:?Set verified Python}"
state=$1
selection=$2
protocol=$3
catalogues=$4
sas=$5
legacy=$6
previous=${7:-}
nodes=${DEEP_PM_BATCH_NODES:-2}
[[ "$nodes" =~ ^[1-9][0-9]*$ ]]
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg
phase=prepare
mkdir -p "$state/logs"
journal() {
    "$DEEP_PM_PYTHON" - "$state" "$phase" "$1" <<'PY'
import json,os,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
event=dict(phase=sys.argv[2],state=sys.argv[3],utc=datetime.now(timezone.utc).isoformat(),
           nodes=int(os.environ.get('DEEP_PM_BATCH_NODES','2')),m3_workers=int(os.environ.get('DEEP_PM_M3_WORKERS','8')),
           streams={k:os.environ.get(k,'2') for k in ('DEEP_PM_FETCH_STREAMS','DEEP_PM_REFERENCE_STREAMS','DEEP_PM_FIT_STREAMS')},
           slurm_job=os.environ['SLURM_JOB_ID'],commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
with (Path(sys.argv[1])/'phase_events.jsonl').open('a') as stream:stream.write(json.dumps(event)+'\n')
PY
}
trap 'code=$?; if [[ "$code" != 0 ]]; then journal "failed_exit_$code"; fi' EXIT
journal started
reuse=()
if [[ -n "$previous" ]]; then reuse=(--previous "$previous"); fi
"$DEEP_PM_PYTHON" -um deep_pm.independent_batch "$state" "$selection" "$protocol" "${reuse[@]}"
journal completed
phase=legacy
journal started
legacy_pid=''
if [[ ! -L "$state/catalogues" ]]; then
    "$DEEP_PM_PYTHON" -um deep_pm.legacy_regions "$state/catalogues" "$legacy" "$protocol" >"$state/logs/legacy.out" 2>"$state/logs/legacy.err" &
    legacy_pid=$!
fi
phase=fetch
journal started
env -u SLURM_CPUS_PER_TASK srun --overlap -N"$nodes" -n"$nodes" -c16 --cpu-bind=none bash scripts/independent_batch_worker.sh "$state" "$catalogues" "$sas" fetch
journal completed
phase=references
journal started
env -u SLURM_CPUS_PER_TASK srun --overlap -N"$nodes" -n"$nodes" -c16 --cpu-bind=none bash scripts/independent_batch_worker.sh "$state" "$catalogues" "$sas" references
journal completed
phase=solve
journal started
env -u SLURM_CPUS_PER_TASK srun --overlap -N"$nodes" -n"$nodes" -c16 --cpu-bind=none bash scripts/independent_batch_solve.sh "$state"
journal completed
phase=legacy
if [[ -n "$legacy_pid" ]]; then wait "$legacy_pid"; fi
journal completed
phase=fit
journal started
env -u SLURM_CPUS_PER_TASK srun --overlap -N"$nodes" -n"$nodes" -c16 --cpu-bind=none bash scripts/independent_batch_fit.sh "$state" "$state/catalogues"
journal completed
