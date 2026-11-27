"""Explain verified empty DR17 inputs using published rows and their within-run flags."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path

from astropy.io import fits
import numpy as np

from .exposure_catalogues import inside
from .footprint_early_coverage_v24 import file_counts, read
from .footprint_intake_v24 import SETTINGS
from .io import digest, save_json
from .sdss_sas import photoobj_path, sdss_field_keys


def classify(fields, file_rows, raw_inside, primary_inside):
    if primary_inside:
        return "requires_scoped_replay"
    if raw_inside:
        return "only_non_primary_within_run"
    if file_rows:
        return "no_published_positions_in_rectangle_and_halo"
    return "empty_candidate_fields" if fields else "no_candidate_field"


def published_rows(hdu):
    if "NAXIS2" in hdu.header:
        return int(hdu.header["NAXIS2"])
    if hdu.header["NAXIS"] == 0 and hdu.data is None:
        return 0
    raise ValueError("Unrecognized SDSS science extension")


def audit(work, sas, union, workers):
    registration = work / "registration"
    path = registration / "private_full_footprint_plan.json"
    receipt = read(union)
    if receipt["signed_footprint"] != digest(path) or receipt["verified"] != receipt["registered"]:
        raise ValueError("Incomplete or changed DR17 scope")
    cells = read(path)["new"]
    halo = SETTINGS["halo_deg"]

    def evidence(name):
        staged = work / "sdss-stage" / name
        proof = staged / "private_sdss.json"
        if proof.is_file():
            early = read(proof)
            if early["rows"]:
                return None
            directory, checksum = staged, early["csv_sha256"]
            keys = early["fields"]
        else:
            pointer = work / "resolved" / (name + ".json")
            directory = Path(read(pointer)["source"]) if pointer.is_file() else work / "intake" / name
            acq = read(directory / "acquisition_summary.json")
            if acq["sdss_measures"]:
                return None
            stage = read(directory / "run_manifest.json")["stages"]["v24-all-measurements"][-1]
            checksum = stage["outputs"]["sdss.csv"]
            spec = cells[name]
            center = tuple(np.mean(spec[k]) for k in ("ra", "dec"))
            radius = np.hypot(np.ptp(spec["ra"])/2, np.ptp(spec["dec"])/2) + 2*halo
            keys = sdss_field_keys(sas, center, radius)
            if len(keys) != acq["sdss_fields"]:
                raise ValueError("Changed set of candidate DR17 fields")
        if file_counts(directory / "sdss.csv")[:2] != (checksum, 0):
            raise ValueError("Empty input is no longer reproducible")
        file_rows = raw_inside = primary_inside = empty_hdus = 0
        for key in keys:
            with fits.open(photoobj_path(sas, key), memmap=True) as hdus:
                count = published_rows(hdus[1])
                empty_hdus += count == 0
                file_rows += count
                if count:
                    data = hdus[1].data
                    mask = inside(data["RA"], data["DEC"], cells[name], halo)
                    raw_inside += np.count_nonzero(mask)
                    primary_inside += np.count_nonzero(mask & ((data["RESOLVE_STATUS"] & 1) != 0))
        return dict(core=name, fields=len(keys), empty_hdus=empty_hdus, file_rows=file_rows, inside=int(raw_inside),
                    primary_inside=int(primary_inside), time_area=cells[name]["time_area"],
                    reason=classify(len(keys), file_rows, raw_inside, primary_inside))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        zeros = [r for r in pool.map(evidence, cells) if r is not None]
    if len(zeros) != receipt["zero_early_rows"]:
        raise ValueError("The empty-input denominator changed")
    job = os.environ["SLURM_JOB_ID"]
    save_json(registration / f"private_empty_sdss_{job}.json", zeros)
    reasons = Counter(r["reason"] for r in zeros)
    output = dict(slurm_job=job, full_early_coverage_sha256=digest(union),
                  signed_footprint=receipt["signed_footprint"], registered=len(cells),
                  empty=len(zeros), reasons=dict(reasons), field_visits=sum(r["fields"] for r in zeros),
                  empty_extension_visits=sum(r["empty_hdus"] for r in zeros),
                  original_positions_with_halo=sum(r["inside"] for r in zeros),
                  run_primary_positions_with_halo=sum(r["primary_inside"] for r in zeros),
                  time_opportunity_sqdeg={k: sum(r["time_area"] for r in zeros if r["reason"]==k)
                                           for k in reasons})
    save_json(registration / f"v24_empty_sdss_{job}_summary.json", output)
    print(output, flush=True)
    if reasons["requires_scoped_replay"]:
        raise ValueError("Verified zero input contains a qualifying published position")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("work", "sas", "union"):
        parser.add_argument(key, type=Path)
    parser.add_argument("--workers", type=int, required=True)
    args = parser.parse_args()
    if not os.getenv("SLURM_JOB_ID") or not 1 <= args.workers <= int(os.environ["SLURM_CPUS_PER_TASK"]):
        parser.error("Use allocated Slurm CPUs")
    audit(**vars(args))
