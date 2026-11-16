#!/usr/bin/env bash
set -euo pipefail
remote=${DEEP_PM_SSH_TARGET:?set the NERSC SSH destination}
project=${DEEP_PM_REMOTE_PROJECT:?set the NERSC project directory}
scratch=${DEEP_PM_REMOTE_RUN:?set the versioned Scratch run directory}
revision=${DEEP_PM_EXPORT_REVISION:?set the deployed exporter commit}
if [[ $# != 2 || ! $remote =~ ^[A-Za-z0-9_.@-]+$ ||
      ! $project =~ ^/[A-Za-z0-9_./-]+$ ||
      ! $scratch =~ ^/[A-Za-z0-9_./-]+$ ||
      ! $revision =~ ^[a-f0-9]{40}$ ]]; then
    exit 2
fi
kind=$1
region=$2
case "$kind" in
    association|precision|matched|prediction|injection|stress|challenge-design|challenge-results) ;;
    *) exit 2 ;;
esac
[[ $region =~ ^[a-z][a-z0-9_]{0,63}$ ]] || exit 2
folder=$(cd "$(dirname "$0")/.." && pwd)
stage=$(mktemp -d)
trap 'if test -f "$stage/result.json"; then unlink "$stage/result.json"; fi; rmdir "$stage"' EXIT
ssh -T -o BatchMode=yes "$remote" "bash -s -- '$project' '$scratch' '$revision' '$kind' '$region'" >"$stage/result.json" <<'REMOTE'
set -euo pipefail
project=$1
export PYTHONPATH="$project/.versions/$3/src:$project/.runtime"
"${DEEP_PM_PYTHON:-$HOME/.conda/envs/desi_env/bin/python}" -m deep_pm.public_summary export "$4" "$5" "$2"
REMOTE
PYTHONPATH="$folder/src" "${DEEP_PM_LOCAL_PYTHON:-python3}" -m deep_pm.public_summary \
    accept "$kind" "$region" "$folder/results/summary" <"$stage/result.json"
