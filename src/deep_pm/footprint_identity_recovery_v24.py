"""Recover the frozen exposure-number population with original raw archive measurements."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from .continuous_reference import verify_directory
from .footprint_intake_v24 import one
from .io import digest, save_json
from .nsc_measurement_identity import measurement_ids


def read(path):
    return json.loads(path.read_text())


def binding(source, shard):
    return dict(source=str(source), shard=str(shard), source_sha256=digest(source / "run_manifest.json"),
                reference_sha256=digest(shard / "v22_reference_input_core_summary.json"))


def validate(source):
    verify_directory(source, ["nsc_object.csv", "detections.csv", "external_labels.csv"])
    native = pd.read_csv(source / "nsc_object.csv", low_memory=False)
    keys = "NSC:" + measurement_ids(native)
    if keys.duplicated().any():
        raise ValueError("Repeated physical ID in the signed raw measurements")
    data = pd.read_csv(source / "detections.csv", low_memory=False)
    nsc = data.loc[data.survey.eq("NSC")].set_index("detection_id", verify_integrity=True)
    if len(nsc) != len(keys) or not keys.isin(nsc.index).all():
        raise ValueError("Normalized NSC identities did not retain every raw physical measurement")
    actual = nsc.loc[keys]
    errors = np.abs(actual[["ra_deg", "dec_deg"]].to_numpy() - native[["ra", "dec"]].to_numpy())
    if not np.isfinite(errors).all() or (errors > 2e-12).any():
        raise ValueError("Physical positions changed in the normalized source")
    if not np.array_equal(actual.exposure_id, "NSC:" + native.exposure):
        raise ValueError("Physical exposure changed in the normalized source")
    labels = pd.read_csv(source / "external_labels.csv", dtype=str).set_index("detection_id", verify_integrity=True)
    if not np.array_equal(labels.loc[keys].original_object_id, native.objectid.astype(str)):
        raise ValueError("Archive label no longer refers to its original physical detection")
    return dict(raw_nsc=len(native), normalized=len(data), unique_raw_ids=native.measid.nunique(),
                qualified_rows=int(keys.str.contains("@", regex=False).sum()),
                max_coordinate_csv_difference_deg=float(errors.max()) if len(errors) else None)


def compare_prior(previous, current):
    # Use the immutable native fields to identify the old record. Its old file
    # was signed with unique raw measid; validate that fact before changing labels.
    a = pd.read_csv(previous / "nsc_object.csv", low_memory=False)
    b = pd.read_csv(current / "nsc_object.csv", low_memory=False)
    order = ["measid", "exposure", "ccdnum"]
    pd.testing.assert_frame_equal(a.sort_values(order).reset_index(drop=True), b.sort_values(order).reset_index(drop=True),
                                  check_dtype=False, check_exact=True)
    if a.measid.duplicated().any():
        raise ValueError("Previously signed input requires a separate membership audit")
    newids = measurement_ids(a)
    changes = newids.ne(a.measid)
    aliases = dict(zip("NSC:" + a.measid[changes], "NSC:" + newids[changes]))
    for name in ["detections.csv", "external_labels.csv"]:
        old = pd.read_csv(previous / name, low_memory=False)
        new = pd.read_csv(current / name, low_memory=False)
        replacements = old.detection_id.map(aliases)
        changed = replacements.notna()
        old.loc[changed, "detection_id"] = replacements.loc[changed]
        pd.testing.assert_frame_equal(old.set_index("detection_id", verify_integrity=True).sort_index(),
                                      new.set_index("detection_id", verify_integrity=True).sort_index(),
                                      check_dtype=False, check_exact=True)


def recover(task):
    name, spec, base, catalogues, sas, fp, rule_sha, revision = task
    work = base / "v2.4"
    audit = work / "physical-identity-audit" / name
    audit.mkdir(exist_ok=True, parents=True)
    control = work / "batch-receipts"
    marker = control / (name + ".json")
    pointer = work / "resolved" / (name + ".json")
    with (control / (name + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        signed = audit / "verified.json"
        if signed.exists():
            old = read(signed)
            if old["rule_sha256"] != rule_sha or old["batch_sha256"] != digest(marker):
                raise ValueError("Completed identity replay changed")
            return old
        pre = audit / "before.json"
        if not pre.exists():
            if marker.is_file():
                state = read(pointer) if pointer.is_file() else binding(work / "intake" / name, work / "input" / name)
                current = read(marker)
                if current["manifest_sha256"] != fp or state["source_sha256"] != current["input_receipt_sha256"]:
                    raise ValueError("Predecessor input binding failed")
                save_json(pre, dict(source=state, batch=current))
            else:
                save_json(pre, dict(source=None, batch=None))
        previous = read(pre)
        prior = previous["source"]
        old_source = Path(prior["source"]) if prior else None
        accepted = False
        if prior:
            if prior != binding(old_source, Path(prior["shard"])):
                raise ValueError("Prior source/reference changed")
            config = (old_source / "config_snapshot.toml").read_text()
            accepted = f'nsc_measurement_identity_sha256 = "{rule_sha}"' in config
        if accepted:
            state, result = prior, previous["batch"]
            fact = validate(old_source)
        else:
            result = one((name, spec, base, catalogues, sas, fp), revision, publish_revision=False)
            folder = work / "revisions" / revision
            source = folder / "intake" / name
            state = binding(source, folder / "input" / name)
            fact = validate(source)
            if prior:
                compare_prior(old_source, source)
            result.update(manifest_sha256=fp, input_receipt_sha256=state["source_sha256"],
                          reference_receipt_sha256=state["reference_sha256"])
            save_json(pointer, state)
            save_json(marker, result)
        proof = dict(**fact, rule_sha256=rule_sha, footprint=fp, batch_sha256=digest(marker),
                     source_sha256=state["source_sha256"], reference_sha256=state["reference_sha256"],
                     previous_signed=prior is not None, compared_previous=bool(prior and not accepted),
                     reference_measurements=result["measurements"])
        save_json(signed, proof)
        print("PHYSICAL_INPUT", name, fact["raw_nsc"], fact["qualified_rows"], flush=True)
        return proof


def run(base, catalogues, sas, workers):
    w = base / "v2.4"
    reg = w / "registration"
    summary = read(reg / "v24_exposure_identity_scope_summary.json")
    path = reg / "private_exposure_identity_scope.json"
    if digest(path) != summary["scope_sha256"] or summary["candidate_old"] != 0:
        raise ValueError("Exposure-scope changed or an old core needs its own frozen-input replay")
    manifest = reg / "private_full_footprint_plan.json"
    fp = digest(manifest)
    rule = os.environ["DPM_NSC_IDENTITY_SHA256"]
    if summary["footprint"] != fp or summary["rule_sha256"] != rule:
        raise ValueError("Input footprint or exposure rule changed")
    cells = read(manifest)["new"]
    scope = read(path)["cores"]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    rev = "exposure-number-" + commit[:12] + "-" + rule[:12]
    # Work on previous receipts and small isolated failures first.
    scope.sort(key=lambda n: (not (w / "batch-receipts" / (n + ".json")).exists(),
                              not (w / "batch-receipts" / (n + ".failed.json")).exists(), n))
    tasks = [(n, cells[n], base, catalogues, sas, fp, rule, rev) for n in scope]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        output = list(pool.map(recover, tasks))
    save_json(reg / "v24_exposure_identity_completion_summary.json",
              dict(footprint=fp, rule_sha256=rule, scope_sha256=digest(path),
                   cores=len(output), cores_with_affected_rows=sum(x["qualified_rows"] > 0 for x in output),
                   previous_sources_replayed=sum(x["compared_previous"] for x in output),
                   raw_nsc_with_halo=sum(x["raw_nsc"] for x in output),
                   qualified_with_halo=sum(x["qualified_rows"] for x in output),
                   normalized_with_halo=sum(x["normalized"] for x in output),
                   reference_with_halo=sum(x["reference_measurements"] for x in output),
                   slurm_job=os.environ["SLURM_JOB_ID"], input_commit=commit))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ["base", "catalogues", "sas"]:
        p.add_argument(key, type=Path)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    if not os.getenv("SLURM_JOB_ID") or not 1 <= a.workers <= int(os.environ["SLURM_CPUS_PER_TASK"]):
        p.error("Use the assigned Slurm CPUs")
    run(a.base, a.catalogues, a.sas, a.workers)
