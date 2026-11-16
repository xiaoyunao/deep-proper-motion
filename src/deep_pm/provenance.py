"""A cached analysis directory belongs to one configuration and catalogue snapshot."""
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
import subprocess

from .io import digest, save_json


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def hash_files(work, filenames):
    return {name: digest(work / name) for name in filenames if (work / name).is_file()}


@contextmanager
def run_stage(work, config_path, region, stage, inputs=(), outputs=()):
    config_hash = digest(config_path)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    manifest = work / "run_manifest.json"
    snapshot = work / "config_snapshot.toml"
    with (work / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if manifest.exists():
            record = json.loads(manifest.read_text())
            if (record.get("commit"), record.get("config_sha256"), record.get("region")) != (
                    commit, config_hash, region):
                raise ValueError("Run directory is bound to another commit, configuration or region")
            if not snapshot.exists() or digest(snapshot) != config_hash:
                raise ValueError("Frozen configuration snapshot is missing or changed")
        else:
            snapshot.write_bytes(config_path.read_bytes())
            record = dict(region=region, commit=commit, config_sha256=config_hash, stages={})
        stages = record["stages"]
        for key, attempts in stages.items():
            last = attempts[-1]
            if last["state"] != "completed":
                continue
            for kind in ("inputs", "outputs"):
                if hash_files(work, last[kind]) != last[kind]:
                    raise ValueError(f"Recorded {kind} changed after {key}")
        attempts = stages.setdefault(stage, [])
        current_inputs = hash_files(work, inputs)
        if attempts and attempts[-1]["state"] == "completed":
            if current_inputs != attempts[-1]["inputs"]:
                raise ValueError(f"Inputs changed after {stage}")
            yield False
            return
        attempt = dict(state="running", started_utc=utc_now(), slurm_job=os.environ.get("SLURM_JOB_ID"),
                       inputs=current_inputs)
        attempts.append(attempt)
        save_json(manifest, record)
        try:
            yield True
            if hash_files(work, inputs) != current_inputs:
                raise ValueError(f"Inputs changed during {stage}")
            attempt.update(state="completed", outputs=hash_files(work, outputs))
        except BaseException as exc:
            attempt.update(state="failed", error_type=type(exc).__name__)
            raise
        finally:
            attempt["finished_utc"] = utc_now()
            save_json(manifest, record)
