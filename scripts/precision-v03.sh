#!/usr/bin/env bash
set -euo pipefail
work=${DEEP_PM_WORK:?use a separate v0.3 Scratch run directory}
source=${DEEP_PM_DETECTION_ROOT:?set the frozen detection root}
published=${DEEP_PM_V02_PRECISION_ROOT:?set the frozen published-motion cache root}
export PYTHONPATH="$PWD/src:${DEEP_PM_EXTRA_PYTHONPATH:-}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
for region in "$@"; do
    input="$source/$region"
    output="$work/$region"
    mkdir -p "$output/validation"
    for name in detections.csv external_labels.csv validation/gaia.csv; do
        test -s "$input/$name"
        if [[ ! -e "$output/$name" ]]; then ln -s "$input/$name" "$output/$name"; fi
        test "$output/$name" -ef "$input/$name"
    done
    if [[ ! -e "$output/cache" ]]; then cp -Rp "$published/$region/cache" "$output/cache"; fi
    "${DEEP_PM_PYTHON:-python}" -u -m deep_pm.precision_v03 "$output" --region "$region"
done
