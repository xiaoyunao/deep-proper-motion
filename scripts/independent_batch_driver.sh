#!/bin/bash
# Two allocated CPU nodes, two independent download/fit streams per node.
set -euo pipefail
: "${SLURM_JOB_ID:?Run inside Slurm}"
: "${DEEP_PM_PYTHON:?Set verified Python}"
state=$1
selection=$2
protocol=$3
catalogues=$4
sas=$5
legacy=$6
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg
phase=prepare
mkdir -p "$state/logs"
journal() {
    "$DEEP_PM_PYTHON" - "$state" "$phase" "$1" <<'PY'
import json,os,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
event=dict(phase=sys.argv[2],state=sys.argv[3],utc=datetime.now(timezone.utc).isoformat(),
           slurm_job=os.environ['SLURM_JOB_ID'],commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
with (Path(sys.argv[1])/'phase_events.jsonl').open('a') as stream:stream.write(json.dumps(event)+'\n')
PY
}
trap 'code=$?; if [[ "$code" != 0 ]]; then journal "failed_exit_$code"; fi' EXIT
journal started
"$DEEP_PM_PYTHON" -um deep_pm.independent_batch "$state" "$selection" "$protocol"
journal completed
phase=legacy
journal started
"$DEEP_PM_PYTHON" -um deep_pm.legacy_regions "$state/catalogues" "$legacy" "$protocol" >"$state/logs/legacy.out" 2>"$state/logs/legacy.err" &
legacy_pid=$!
phase=fetch
journal started
srun --overlap -N2 -n2 -c16 --cpu-bind=none bash scripts/independent_batch_worker.sh "$state" "$catalogues" "$sas" fetch
journal completed
phase=references
journal started
srun --overlap -N2 -n2 -c16 --cpu-bind=none bash scripts/independent_batch_worker.sh "$state" "$catalogues" "$sas" references
journal completed
phase=solve
journal started
srun --overlap -N2 -n2 -c16 --cpu-bind=none bash scripts/independent_batch_solve.sh "$state"
journal completed
phase=legacy
wait "$legacy_pid"
journal completed
phase=fit
journal started
srun --overlap -N2 -n2 -c16 --cpu-bind=none bash scripts/independent_batch_fit.sh "$state" "$state/catalogues"
journal completed
