#!/bin/bash
# Arrange the independent quality readout after the owner and blind products close.
set -euo pipefail
mode=$1
work=$2
code=$3
account=$4
python=$5
runtime=$6
previous=$7
previous_blind=$8
spectra=$9
export PYTHONPATH="$code/src:$runtime" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$code"
case "$mode" in
  dispatch)
    ledger="$work/private_production_jobs.json"
    upstream=$("$python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["jobs"]["science"]["job"])' "$ledger")
    [[ "$upstream" =~ ^[0-9]+$ ]]
    [[ ! -e "$work/post-readout-job-id" ]]
    sbatch --parsable -A "$account" -q regular -C cpu -N1 -n1 -c12 --mem=32G -t 00:30:00 \
      --job-name=dpm-v19-replay-readout --kill-on-invalid-dep=yes --dependency="afterok:$upstream" \
      --chdir="$code" --output="$work/logs/replay-readout-%j.out" "$code/scripts/wide-v19-followup.sh" \
      run "$work" "$code" "$account" "$python" "$runtime" "$previous" "$previous_blind" "$spectra" \
      | tee "$work/post-readout-job-id"
    ;;
  run)
    "$python" -um deep_pm.production_replay_readout "$previous" "$previous_blind" "$work" \
      "$work/science/v19_production_replay_quality_summary.json" --spectra-file "$spectra"
    ;;
  *) printf 'Unknown stage: %s\n' "$mode" >&2; exit 2 ;;
esac
