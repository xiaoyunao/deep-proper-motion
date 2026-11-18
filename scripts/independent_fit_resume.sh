#!/bin/bash
# Refit unchanged C0 from verified completed acquisition and common graphs.
set -euo pipefail
: "${SLURM_JOB_ID:?Run inside Slurm}"
: "${DEEP_PM_PYTHON:?Set verified Python}"
state=$1
previous=$2
protocol=$3
nodes=${DEEP_PM_BATCH_NODES:-2}
[[ "$nodes" =~ ^[1-9][0-9]*$ ]]
mkdir -p "$state/logs"
"$DEEP_PM_PYTHON" -um deep_pm.independent_batch "$state" "$previous/selection.json" "$protocol" --previous "$previous"
"$DEEP_PM_PYTHON" - "$state" "$previous" <<'PY'
import json,sys
from pathlib import Path
from deep_pm.continuous_reference import verify_directory
from deep_pm.io import digest,save_json
state,previous=map(Path,sys.argv[1:])
receipts={}
for box in sorted({line.split()[1] for line in (state/'tasks.tsv').read_text().splitlines()}):
    if (state/f'batch-{box}.toml').read_bytes()!=(previous/f'batch-{box}.toml').read_bytes():
        raise ValueError('Graph reuse requires exactly the same cumulative geometry and projection')
    target=previous/'shared'/box
    receipt=verify_directory(target,('private_c0.npz','continuous_reference_summary.json'))
    link=state/'shared'/box
    link.parent.mkdir(parents=True,exist_ok=True)
    if not link.exists():link.symlink_to(target.resolve(strict=True))
    if link.resolve()!=target.resolve():raise ValueError('Reference graph target changed')
    receipts[box]=dict(receipt_sha256=digest(receipt),solution_sha256=digest(target/'private_c0.npz'))
save_json(state/'reference_reuse_receipt.json',dict(graphs=receipts,unchanged_cumulative_geometry=True))
PY
date -u +"C0 fits started %FT%TZ"
env -u SLURM_CPUS_PER_TASK srun --overlap -N"$nodes" -n"$nodes" -c16 --cpu-bind=none bash scripts/independent_batch_fit.sh "$state" "$state/catalogues"
date -u +"C0 fits completed %FT%TZ"
"$DEEP_PM_PYTHON" -um deep_pm.independent_summary "$state" "$protocol" "$state/unsealed_summary.json"
