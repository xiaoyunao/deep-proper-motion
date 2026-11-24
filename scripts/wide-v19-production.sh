#!/bin/bash
# Batch steps use explicit paths and the same Git snapshot on every compute node.
set -euo pipefail
mode=$1
work=$2
code=$3
account=$4
qos=$5
python=$6
runtime=$7
shift 7
mkdir -p "$work/logs"
cd "$code"
export DPM_COMMIT
DPM_COMMIT=$(git rev-parse HEAD)
export PYTHONPATH="$code/src:$runtime" OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
registered="$work/registered"
case "$mode" in
  intake)
    frozen=$1
    catalogues=$2
    sas=$3
    legacy=$4
    window=$5
    if [[ -n ${DPM_SIGNED_INTAKE_DRIVER:-} ]]; then
      "$python" -u "$DPM_SIGNED_INTAKE_DRIVER" "$work/intake" "$registered" "$frozen" "$catalogues" "$sas" "$legacy"
    else
      "$python" -um deep_pm.production_intake "$work/intake" "$registered" "$frozen" "$catalogues" "$sas" "$legacy"
    fi
    "$python" -um deep_pm.production_reference_plan "$work/reference" "$work/intake" "$registered" "$frozen"
    "$python" -um deep_pm.production_dispatch "$work" "$code" "$account" "$qos" "$python" "$runtime" "$window"
    ;;
  reference)
    label=$1
    root="$work/reference"
    domain="$root/$label"
    config="$root/private_continuation_reference_plan.json"
    mapfile -t metadata < <("$python" -c '
import json,sys
from pathlib import Path
from deep_pm.io import digest
root=Path(sys.argv[1])
path=root/"private_continuation_reference_plan.json"
proof=json.loads((root/"v19_reference_plan_summary.json").read_text())
if digest(path)!=proof["plan_sha256"] or proof["forecast_cpu_node_hours"]>proof["budget_cpu_node_hours"]:
    raise ValueError("Changed or over-budget reference plan")
domain=json.loads(path.read_text())["solve"][sys.argv[2]]
print(*domain["center"],sep="\n")
print(*domain["windows"],sep="\n")
' "$root" "$label")
    if (( ${#metadata[@]} < 3 )); then exit 1; fi
    "$python" -um deep_pm.joint_reference "$domain" "$domain/layout" --center "${metadata[0]}" "${metadata[1]}" --boxes "${metadata[@]:2}" --workers 12
    export DEEP_PM_M3_WORKERS=3
    "$python" -um deep_pm.continuous_reference solve "$domain/shared" "$domain/joint_geometry.toml" "$label" \
      "$domain/private_joint_reference_plan.json" --separate-components --checkpoint-dir "$domain/checkpoints"
    ;;
  fit)
    "$python" -um deep_pm.production_fit "$work/fit" "$work/reference" "$work/intake" 6
    ;;
  science)
    "$python" -um deep_pm.production_science "$work/science" "$work/reference" "$work/fit" "$work/intake" "$1"
    "$python" -um deep_pm.blind_catalogue_quality "$work/science" "$work/blind_quality" --workers 16
    ;;
  *) printf 'Unknown stage: %s\n' "$mode" >&2; exit 2 ;;
esac
