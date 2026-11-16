#!/usr/bin/env bash
set -euo pipefail
remote=${DEEP_PM_SSH_TARGET:?Set DEEP_PM_SSH_TARGET}
source_root=${DEEP_PM_REMOTE_SCRATCH:?Set DEEP_PM_REMOTE_SCRATCH to the remote experiment directory}
[[ $source_root =~ ^/[A-Za-z0-9_./-]+$ && $remote =~ ^[A-Za-z0-9_.@-]+$ ]] || exit 2
folder=$(cd "$(dirname "$0")/.." && pwd)
stage=$(mktemp -d)
trap 'rm -rf -- "$stage"' EXIT
ssh -T -o BatchMode=yes "$remote" "bash -s -- '$source_root'" >"$stage/summary.tar" <<'REMOTE'
set -euo pipefail
cd "$1"
allowed=()
for region in smoke wrap north pal5 challenge; do
    for name in route_audit_summary counterfactual_summary diagnostics_summary injection_summary \
        solver_ablation_summary night_bootstrap_summary exhaustive_summary; do
        path="$region/$name.json"
        if [[ -f $path && ! -L $path ]]; then allowed+=("$path"); fi
    done
    path="$region/summary/${region}_summary.json"
    if [[ -f $path && ! -L $path ]]; then allowed+=("$path"); fi
    for plot in epochs recovery paired residuals cases coverage; do
        path="$region/summary/${region}_${plot}_summary.png"
        if [[ -f $path && ! -L $path ]]; then allowed+=("$path"); fi
    done
done
tar -cf - -- "${allowed[@]}"
REMOTE
tar -xf "$stage/summary.tar" -C "$stage"
mkdir -p "$folder/figures" "$folder/results/summary"
shopt -s nullglob
for region in smoke wrap north pal5 challenge; do
    for f in "$stage/$region/"*_summary.json; do
        python3 -m json.tool "$f" >/dev/null
        dest="$folder/results/summary/${region}_${f##*/}"
        if [[ ${f##*/} == diagnostics_summary.json ]]; then
            python3 - "$f" "$dest" <<'PYTHON'
import json
from pathlib import Path
import sys

summary = json.loads(Path(sys.argv[1]).read_text())
summary.pop("cases", None)
Path(sys.argv[2]).write_text(json.dumps(summary, indent=2, allow_nan=False) + "
")
PYTHON
        else
            cp -- "$f" "$dest"
        fi
    done
    for f in "$stage/$region/summary/"*_summary.json; do
        python3 -m json.tool "$f" >/dev/null
        cp -- "$f" "$folder/results/summary/"
    done
    for f in "$stage/$region/summary/"*_summary.png; do
        cp -- "$f" "$folder/figures/"
    done
done
