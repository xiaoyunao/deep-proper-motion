#!/usr/bin/env bash
# Run inside an allocated Perlmutter node from a versioned worktree.
set -euo pipefail
mode=${1:?plan, solve, fit or summarize}
state=${2:?private run directory}
config=${3:?frozen input configuration}
legacy=${4:?Xiao regional CSV}
: "${SLURM_JOB_ID:?run on a Slurm node}"
python=${DEEP_PM_PYTHON:-python3}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "$state"
if [[ "$mode" == plan ]]; then
    mkdir -p "$state/refs"
    "$python" - "$state" <<'PY'
import json,sys
from pathlib import Path
state=Path(sys.argv[1])
refs, sources = {}, {}
for line in (state/"run-plan.tsv").read_text().splitlines():
    name,source,ref=line.split("\t")
    receipt=Path(ref)/"run_manifest.json"
    if not receipt.is_file():
        continue
    stage=json.loads(receipt.read_text())["stages"].get("continuous-ref-input-v12", [])
    if stage and stage[-1]["state"]=="completed":
        refs[name],sources[name]=ref,source
(state/"references.json").write_text(json.dumps(refs))
(state/"sources.json").write_text(json.dumps(sources))
if not refs:
    raise ValueError("No complete reference inputs")
PY
elif [[ "$mode" == solve ]]; then
    "$python" -um deep_pm.continuous_reference solve "$state/shared" "$config" whole_c "$state/references.json" --separate-components
elif [[ "$mode" == fit ]]; then
    : "${SLURM_ARRAY_TASK_ID:?requires a Slurm job array}"
    IFS=$'\t' read -r region src ref < <(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$state/run-plan.tsv")
    if [[ ! -f "$ref/run_manifest.json" ]]; then
        printf 'Missing complete input for %s\n' "$region"
        exit 0
    fi
    if ! "$python" -c 'import json,sys; raise SystemExit(sys.argv[1] not in json.load(open(sys.argv[2])))' "$region" "$state/sources.json"; then
        printf 'Incomplete input for %s\n' "$region"
        exit 0
    fi
    dst="$state/outputs/$region"
    mkdir -p "$dst"
    cache="${DEEP_PM_GAIA_QUALITY_ROOT:-}/$region/gaia_quality_cache"
    if [[ -d "$cache" && ! -e "$dst/gaia_quality_cache" ]]; then
        ln -s "$cache" "$dst/gaia_quality_cache"
    fi
    "$python" -um deep_pm.continuous_fit "$dst" "$region" "$src" "$state/shared" "$legacy" "$config" --annual
elif [[ "$mode" == summarize ]]; then
    "$python" -um deep_pm.continuous_coverage "$state" "$config" --prepare-summary
    "$python" -um deep_pm.continuous_summary "$state/v12_full_continuous_summary.json" "$state/fits.json" "$state/gauges.json" "$config"
else
    printf 'Unknown continuous run stage %s\n' "$mode" >&2
    exit 2
fi
