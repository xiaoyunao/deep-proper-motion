#!/bin/bash
set -euo pipefail
: "${SLURM_JOB_ID:?Run inside Slurm}"
: "${DEEP_PM_PYTHON:?Set the verified Python executable}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg
frozen=$1
output=$2
legacy=$3
config=$4
"$DEEP_PM_PYTHON" -um deep_pm.continuous_condition_pool "$frozen" "$output" "$output/canary_summary.json"
"$DEEP_PM_PYTHON" - "$frozen/run-plan.tsv" "$output" <<'PY' > "$output/pending-task-indices.txt"
import json,sys
from pathlib import Path
for i,line in enumerate(Path(sys.argv[1]).read_text().splitlines()):
    name=line.split('\t')[0]
    manifest=Path(sys.argv[2])/name/'run_manifest.json'
    if manifest.is_file():
        stages=json.loads(manifest.read_text())['stages'].get('continuous-conditions-v13',[])
        if stages and stages[-1]['state']=='completed':
            continue
    print(i)
PY
xargs -r -P"${DEEP_PM_PARALLEL:-8}" -I{} env SLURM_ARRAY_TASK_ID={} bash scripts/continuous_conditions_batch.sh "$frozen" "$output" "$legacy" "$config" < "$output/pending-task-indices.txt"
"$DEEP_PM_PYTHON" -um deep_pm.continuous_condition_pool "$frozen" "$output" "$output/full_conditions_summary.json"
