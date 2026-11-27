"""Count each new registered core once and verify its available early-position bytes."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
import os
from pathlib import Path

from .io import digest, save_json


def read(path):
    return json.loads(path.read_text())


def file_counts(path):
    checksum = sha256()
    lines = size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
            lines += block.count(b"\n")
            size += len(block)
    return checksum.hexdigest(), lines - 1, size


def inspect_core(work, name, spec, footprint, reader):
    staged = work / "sdss-stage" / name
    proof = staged / "private_sdss.json"
    receipt = work / "batch-receipts" / (name + ".json")
    alternatives = {}
    if proof.is_file():
        early = read(proof)
        binding = early["binding"]
        if (early["footprint"] != footprint or binding["reader_sha256"] != reader
                or any(binding[key] != spec[key] for key in ("ra", "dec"))):
            raise ValueError("early binding")
        alternatives["early"] = (staged / "sdss.csv", early["csv_sha256"], early["rows"])
    if receipt.is_file():
        core = read(receipt)
        if core["manifest_sha256"] != footprint or core["core"] != name:
            raise ValueError("dual footprint")
        pointer = work / "resolved" / (name + ".json")
        source = Path(read(pointer)["source"]) if pointer.is_file() else work / "intake" / name
        manifest = source / "run_manifest.json"
        if digest(manifest) != core["input_receipt_sha256"]:
            raise ValueError("dual manifest")
        stage = read(manifest)["stages"]["v24-all-measurements"][-1]
        acquisition = source / "acquisition_summary.json"
        if stage["state"] != "completed" or digest(acquisition) != stage["outputs"][acquisition.name]:
            raise ValueError("dual acquisition")
        alternatives["dual"] = (source / "sdss.csv", stage["outputs"]["sdss.csv"],
                                read(acquisition)["sdss_measures"])
    if not alternatives:
        return "absent", 0, 0
    if len(set(item[1:] for item in alternatives.values())) != 1:
        raise ValueError("early and dual SDSS disagree")
    path, expected_sha, rows = next(iter(alternatives.values()))
    actual_sha, actual_rows, size = file_counts(path)
    if (actual_sha, actual_rows) != (expected_sha, rows):
        raise ValueError("CSV checksum or rows")
    return "+".join(alternatives), rows, size


def run(work, workers):
    registration = work / "registration"
    plan = registration / "private_full_footprint_plan.json"
    footprint = digest(plan)
    replay = read(registration / "v24_sdss_stage_replay_summary.json")
    if not replay["all_four_byte_equal"] or replay["signed_footprint"] != footprint:
        raise ValueError("Unverified early-position reader")
    cells = read(plan)["new"]

    def inspect(name):
        try:
            return inspect_core(work, name, cells[name], footprint, replay["reader_sha256"])
        except (KeyError, ValueError, OSError) as exc:
            return "unverified", 0, str(exc)

    counts = Counter()
    rows = size = empty = 0
    opportunity = 0.
    missing = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for name, (route, count, value) in zip(cells, pool.map(inspect, cells)):
            counts[route] += 1
            if route in ("absent", "unverified"):
                missing[name] = route if route == "absent" else value
                continue
            empty += count == 0
            rows += count
            size += value
            opportunity += cells[name]["time_area"]
    job = os.environ["SLURM_JOB_ID"]
    save_json(registration / f"private_early_exceptions_{job}.json", missing)
    result = dict(signed_footprint=footprint, reader_sha256=replay["reader_sha256"],
                  slurm_job=job, registered=len(cells), verified=len(cells)-len(missing),
                  by_signed_input=dict(counts), zero_early_rows=empty, rows_with_halo=rows,
                  verified_bytes=size, time_opportunity_sqdeg=opportunity,
                  scope="One registered new core per entry. One CSV per core was read in full. "
                        "Coexisting intake and early-stage receipts must agree on SHA256 and row count. "
                        "Rows include repeated halos. Time opportunity is registered geometry.")
    save_json(registration / f"v24_early_coverage_{job}_summary.json", result)
    print(result, flush=True)
    if missing:
        raise RuntimeError(f"{len(missing)} registered cores lack verified DR17 input")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    parser.add_argument("--workers", required=True, type=int)
    args = parser.parse_args()
    if not os.getenv("SLURM_JOB_ID") or not 1 <= args.workers <= int(os.environ["SLURM_CPUS_PER_TASK"]):
        parser.error("Use allocated Slurm CPUs")
    run(args.work, args.workers)
