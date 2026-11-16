#!/usr/bin/env bash
set -euo pipefail
remote=${DEEP_PM_SSH_TARGET:?Set DEEP_PM_SSH_TARGET}
source_root=${DEEP_PM_REMOTE_SCRATCH:?Set DEEP_PM_REMOTE_SCRATCH to the remote experiment directory}
[[ $source_root =~ ^/[A-Za-z0-9_./-]+$ && $remote =~ ^[A-Za-z0-9_.@-]+$ ]] || exit 2
socket_dir=$(mktemp -d)
socket="$socket_dir/connection"
ssh -o BatchMode=yes -MNf -S "$socket" "$remote"
trap 'ssh -S "$socket" -O exit "$remote" >/dev/null 2>&1 || true; rmdir "$socket_dir"' EXIT
folder=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$folder/figures" "$folder/results/summary"
for region in smoke wrap north pal5; do
    for name in "$region"_summary route_audit_summary counterfactual_summary diagnostics_summary \
        injection_summary solver_ablation_summary night_bootstrap_summary exhaustive_summary; do
        remote_dir="$source_root/$region"
        dest="$folder/results/summary/$region-$name.json"
        if [[ $name == "$region"_summary ]]; then
            remote_dir="$remote_dir/summary"
            dest="$folder/results/summary/$name.json"
        else
            dest="$folder/results/summary/${region}_$name.json"
        fi
        if ssh -S "$socket" -o BatchMode=yes "$remote" "test -s '$remote_dir/$name.json'"; then
            scp -q -o "ControlPath=$socket" "$remote:$remote_dir/$name.json" "$dest.part"
            python3 -m json.tool "$dest.part" >/dev/null
            mv "$dest.part" "$dest"
        fi
    done
    for plot in epochs recovery paired residuals cases; do
        dest="$folder/figures/${region}_${plot}_summary.png"
        rel="$source_root/$region/summary/${region}_${plot}_summary.png"
        if ssh -S "$socket" -o BatchMode=yes "$remote" "test -s '$rel'"; then
            scp -q -o "ControlPath=$socket" "$remote:$rel" "$dest.part"
            mv "$dest.part" "$dest"
        fi
    done
done
