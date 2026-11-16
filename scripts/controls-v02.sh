#!/usr/bin/env bash
set -euo pipefail
work=${DEEP_PM_WORK:?set a separate Scratch directory for v0.2 controls}
source_root=${DEEP_PM_FROZEN_V01:?set the v0.1 Scratch root}
window="${DEEP_PM_SDSS_WINDOW:?set the verified SDSS window_flist FITS path}"
python="${DEEP_PM_PYTHON:-python}"
export PYTHONPATH="$PWD/src:${DEEP_PM_EXTRA_PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
config=configs/experiment-a-v02.toml
if [[ $work == "$source_root" ]]; then exit 2; fi
case "$1" in
    prediction|injection)
        mode=$1
        shift
        for region in "$@"; do
            case "$region" in smoke|wrap|north|pal5) ;; *) exit 2 ;; esac
            region_work="$work/$region"
            mkdir -p "$region_work/validation"
            for rel in detections.csv validation/gaia.csv; do
                if [[ ! -e "$region_work/$rel" ]]; then
                    ln -s "$source_root/$region/$rel" "$region_work/$rel"
                fi
                test -s "$region_work/$rel"
            done
            echo "$region $mode"
            if [[ $mode == prediction ]]; then
                "$python" -u -m deep_pm.counterfactual "$region_work" --region "$region"
            else
                "$python" -u -m deep_pm.injection "$region_work" --region "$region" --sdss-window "$window" --trials 36
            fi
        done ;;
    stress)
        for route in sdss object; do
            "$python" -u -m deep_pm --work "$work" --config "$config" --region stress_north --route "$route" fetch
        done
        "$python" -u -m deep_pm --work "$work" --config "$config" --region stress_north normalize
        "$python" -u -m deep_pm.injection "$work/stress_north" --region stress_north --sdss-window "$window" --trials 36 --stress
        ;;
    *) exit 2 ;;
esac
