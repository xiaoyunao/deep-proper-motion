#!/usr/bin/env bash
# Run inside a Slurm step, one sequential Data Lab download stream per rank.
set -euo pipefail
run_root=${1:?scratch output directory}
list=${2:?file of science core names in dispatch order}
catalogues=${3:?verified optical catalogue directory}
sas=${4:?SDSS data root}
config=${5:?frozen configuration}
: "${SLURM_JOB_ID:?run on a Slurm compute node}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python=${DEEP_PM_PYTHON:-python3}
rank=${SLURM_PROCID:-0}
ranks=${SLURM_NTASKS:-1}
status=0
while IFS= read -r region; do
    [[ $region =~ ^c_[0-9][0-9]_[0-9][0-9]$ ]] || exit 2
    echo "starting $region"
    if "$python" -um deep_pm.exposure_inputs "$run_root" "$region" "$catalogues" "$sas" "$config" --workers 6; then
        echo "completed $region"
    else
        echo "incomplete $region"
        status=1
    fi
done < <(awk -v rank="$rank" -v ranks="$ranks" '(NR-1)%ranks==rank' "$list")
exit "$status"
