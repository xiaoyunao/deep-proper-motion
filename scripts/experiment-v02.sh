#!/usr/bin/env bash
set -euo pipefail
work=${DEEP_PM_WORK:?set a v0.2 Scratch run directory}
cache=${DEEP_PM_FROZEN_V01:?set the existing v0.1 Scratch project directory}
python=${DEEP_PM_PYTHON:-python}
export PYTHONPATH="$PWD/src:${DEEP_PM_EXTRA_PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
if [[ $work == "$cache" ]]; then
    printf '%s\n' 'Select a separate run directory' >&2
    exit 2
fi
for region in "$@"; do
    case "$region" in smoke|wrap|north|pal5) ;; *) exit 2 ;; esac
    mkdir -p "$work/$region/validation"
    for rel in detections.csv validation/gaia.csv external_labels.csv sdss.csv; do
        f="$work/$region/$rel"
        if [[ ! -e $f ]]; then ln -s "$cache/$region/$rel" "$f"; fi
        test -s "$f"
    done
    for stage in inventory associate fit validate report; do
        printf '%s %s\n' "$region" "$stage"
        "$python" -u -m deep_pm --config configs/experiment-a-v02.toml --work "$work" --region "$region" "$stage"
    done
done
