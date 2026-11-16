#!/usr/bin/env bash
set -euo pipefail
work=${DEEP_PM_WORK:?set a versioned Scratch directory}
python=${DEEP_PM_PYTHON:-python}
export PYTHONPATH="$PWD/src:${DEEP_PM_EXTRA_PYTHONPATH:-}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mode=$1
shift
for region in "$@"; do
    output="$work/$region"
    mkdir -p "$output"
    case "$mode" in
        bright)
            start="${DEEP_PM_V03_BRIGHT:?set the frozen v0.3 bright region root}/$region"
            for input in detections.csv external_labels.csv validation cache; do
                test -e "$start/$input"
                if [[ ! -e "$output/$input" ]]; then ln -s "$start/$input" "$output/$input"; fi
                test "$start/$input" -ef "$output/$input"
            done
            "$python" -u -m deep_pm.nsc_comparison "$output" "$region" bright
            ;;
        inventory)
            "$python" -u -m deep_pm.qso_fields "$output" "$region" inventory \
                --catalogue "${DEEP_PM_QSO_FITS:?set the verified DESI DR1 QSO FITS}"
            ;;
        qso)
            "$python" -u -m deep_pm.qso_fields "$output" "$region" fetch
            "$python" -u -m deep_pm.nsc_comparison "$output" "$region" qso
            ;;
        *)
            exit 2
            ;;
    esac
done
