#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?use an allocated compute node}"
inputs=${DEEP_PM_INPUTS:?set the versioned Scratch acquisition root}
outputs=${DEEP_PM_OUTPUTS:?set the versioned Scratch production root}
sas=${DEEP_PM_SAS:?set the verified DR17 SAS}
catalogue=${DEEP_PM_QSO:?set the verified DESI DR1 QSO FITS}
py=${DEEP_PM_PYTHON:-python}
mkdir -p "$outputs"
pids=()
names=()
for region in "$@"; do
    [[ $region =~ ^[a-z][a-z0-9_]+$ ]]
    (
      set -e
      "$py" -u -m deep_pm.regional_inputs "$inputs/$region" "$region" "$catalogue" "$sas"
      "$py" -u -m deep_pm.regional_production "$outputs/$region" "$region" "$inputs/$region"
    ) >"$outputs/$region-${SLURM_JOB_ID}.log" 2>&1 &
    pids+=("$!")
    names+=("$region")
done
failed=0
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
        printf 'Completed %s\n' "${names[$i]}"
    else
        printf 'Failed %s\n' "${names[$i]}" >&2
        failed=1
    fi
done
exit "$failed"
