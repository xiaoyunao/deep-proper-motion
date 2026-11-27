"""Acquire frozen DR17 r inputs independently of the availability of the NSC archive."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import fcntl
from functools import lru_cache
from hashlib import sha256
import inspect
import json
import os
from pathlib import Path
from time import monotonic

import numpy as np

from . import astrometry, sdss_sas
from .access import SDSS_COLUMNS
from .exposure_catalogues import inside
from .io import digest, save_json


def read(path):
    return json.loads(path.read_text())


def stat_key(path):
    s = path.stat()
    return [s.st_size, s.st_mtime_ns]


@lru_cache(maxsize=1)
def reader_hash():
    code = [digest(Path(m.__file__)) for m in (sdss_sas, astrometry)]
    code.extend([inspect.getsource(inside), SDSS_COLUMNS, inspect.getsource(read_early)])
    return sha256(json.dumps(code).encode()).hexdigest()


def read_early(sas, spec, halo):
    center = tuple(np.mean(spec[k]) for k in ("ra", "dec"))
    radius = float(np.hypot(np.ptp(spec["ra"])/2, np.ptp(spec["dec"])/2)) + 2*halo
    keys = sdss_sas.sdss_field_keys(sas, center, radius)
    signature = field_hash(sas, keys)
    data, _ = sdss_sas.read_fields(sas, center, radius, keys, 1)
    clipped = data.loc[inside(data.ra, data.dec, spec, halo)]
    if field_hash(sas, keys) != signature:
        raise ValueError("DR17 inputs changed during field extraction")
    return clipped, keys, signature


def field_hash(sas, keys):
    h = sha256()
    for key in keys:
        h.update(json.dumps([list(key), stat_key(sdss_sas.photoobj_path(sas, key))]).encode())
    return h.hexdigest()


def binding(sas, spec, halo):
    index = sas/"resolve/2013-07-29/window_flist.fits"
    return dict(sas=str(sas.resolve()), index_stat=stat_key(index), ra=list(spec["ra"]),
                dec=list(spec["dec"]), halo=halo, reader_sha256=reader_hash())


def adopt(staged, dest, sas, spec, halo, *, footprint):
    """Copy exact signed bytes. A later legacy rerun can safely rewrite its own file."""
    proof = staged/"private_sdss.json"
    if not proof.is_file():
        return None
    record = read(proof)
    if record["footprint"] != footprint or record["binding"] != binding(sas, spec, halo):
        raise ValueError("Staged SDSS geometry or reader changed")
    if field_hash(sas, record["fields"]) != record["field_sha256"]:
        raise ValueError("Published SDSS FITS inputs changed")
    tmp = dest.with_suffix(".sdss-draft")
    checksum = sha256()
    lines = 0
    with (staged/"sdss.csv").open("rb") as source, tmp.open("wb") as output:
        for block in iter(lambda: source.read(1024*1024), b""):
            output.write(block)
            checksum.update(block)
            lines += block.count(b"\n")
    if checksum.hexdigest() != record["csv_sha256"] or lines != record["rows"] + 1:
        tmp.unlink()
        raise ValueError("Staged DR17 r CSV checksum or row count mismatch")
    tmp.replace(dest)
    return record["rows"], len(record["fields"])


def one(task):
    w, name, spec, sas, halo, footprint, compare = task
    d = w/"sdss-stage"/name
    d.mkdir(exist_ok=True, parents=True)
    with (d/"stage.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        proof = d/"private_sdss.json"
        if not proof.is_file():
            origin = binding(sas, spec, halo)
            data, keys, files_sha = read_early(sas, spec, halo)
            if origin != binding(sas, spec, halo):
                raise ValueError("Changed SDSS field index during staging")
            tmp = d/"sdss.part"
            data.to_csv(tmp, index=False)
            tmp.replace(d/"sdss.csv")
            save_json(proof, dict(binding=origin, footprint=footprint, fields=keys, field_sha256=files_sha,
                                  rows=len(data), csv_sha256=digest(d/"sdss.csv"), job=os.environ["SLURM_JOB_ID"]))
        rec = read(proof)
        if rec["footprint"] != footprint or rec["binding"] != binding(sas, spec, halo):
            raise ValueError("Changed registered SDSS stage")
        if compare:
            pointer = w/"resolved"/(name+".json")
            old = Path(read(pointer)["source"]) if pointer.exists() else w/"intake"/name
            history = read(old/"run_manifest.json")["stages"]["v24-all-measurements"][-1]
            if history["state"] != "completed":
                raise ValueError("Comparison needs a completed frozen source")
            actual = adopt(d, d/"compare.csv", sas, spec, halo, footprint=footprint)
            original = read(old/"acquisition_summary.json")
            expected = (original["sdss_measures"], original["sdss_fields"])
            if (actual != expected or history["outputs"]["sdss.csv"] != digest(d/"compare.csv")
                    or digest(old/"sdss.csv") != rec["csv_sha256"]):
                raise ValueError("DR17 r replay disagrees with the frozen acquisition")
            (d/"compare.csv").unlink()
        return rec["rows"], len(rec["fields"]), (d/"sdss.csv").stat().st_size


def run(work, sas, start, end, workers, pilots):
    from .footprint_intake_v24 import SETTINGS, select_pilots
    from .footprint_reference_packs_v24 import groups
    reg = work/"registration"
    path = reg/"private_full_footprint_plan.json"
    footprint = digest(path)
    if read(reg/"v24_footprint_inventory_summary.json")["signed_manifest_sha256"] != footprint:
        raise ValueError("Changed registered SDSS footprint")
    cells = read(path)["new"]
    if pilots:
        names = select_pilots(cells)
    else:
        admission = read(reg/"v24_sdss_stage_replay_summary.json")
        if not admission["all_four_byte_equal"] or admission["signed_footprint"] != footprint or admission["reader_sha256"] != reader_hash():
            raise ValueError("Missing matching DR17 input replay")
        ordered = [n for group in groups(cells)[1:] for n in group]
        if not 0 <= start < end <= len(ordered):
            raise ValueError("Outside registered input ordinals")
        names = [n for n in ordered[start:end] if not (work/"batch-receipts"/(n+".json")).exists()]
    t = monotonic()
    tasks = [(work, n, cells[n], sas, SETTINGS["halo_deg"], footprint, pilots) for n in names]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        result = list(pool.map(one, tasks))
    job = os.environ["SLURM_JOB_ID"]
    summary = dict(signed_footprint=footprint, reader_sha256=reader_hash(), slurm_job=job,
                   cores=len(result), sdss_detections_with_halo=sum(v[0] for v in result),
                   field_visits_including_overlaps=sum(v[1] for v in result),
                   csv_bytes=sum(v[2] for v in result), elapsed_seconds=monotonic()-t,
                   workers=workers, scope="Signed DR17 r rows before normalization, with overlapping halos.")
    if pilots:
        if len(result) != 4:
            raise ValueError("Missing fixed sky-quadrant control")
        summary["all_four_byte_equal"] = True
        target = reg/"v24_sdss_stage_replay_summary.json"
    else:
        summary["ordinal_range"] = [start, end]
        target = reg/("v24_sdss_stage_"+job+"_summary.json")
    save_json(target, summary)
    print(summary, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    parser.add_argument("sas", type=Path)
    parser.add_argument("--pilots", action="store_true")
    parser.add_argument("--start", type=int)
    parser.add_argument("--end", type=int)
    parser.add_argument("--workers", type=int, required=True)
    args = parser.parse_args()
    if not os.getenv("SLURM_JOB_ID") or not 1 <= args.workers <= int(os.environ["SLURM_CPUS_PER_TASK"]):
        parser.error("Use the allocated Slurm CPUs")
    run(args.work, args.sas, args.start, args.end, args.workers, args.pilots)
