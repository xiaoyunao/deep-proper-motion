"""Acquire complete regional data using the index only for downloads."""
import argparse
import os
from pathlib import Path

import pandas as pd

from . import access
from .__main__ import standardize
from .nsc_comparison import configuration
from .provenance import run_stage


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("work", type=Path)
    p.add_argument("region")
    p.add_argument("coverage", type=Path)
    p.add_argument("--config", type=Path, default=Path("configs/production-v05.toml"))
    args = p.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        p.error("Use a Slurm compute node.")
    _, settings = configuration(args.config)
    field = settings["fields"][args.region]
    center = field["ra"], field["dec"]
    args.work.mkdir(parents=True, exist_ok=True)
    inputs = ("spectroscopic_qso.csv", "sdss.csv", "nsc_object_index.csv", "qso_coverage_summary.json")
    for name in inputs:
        target = (args.coverage/name).resolve(strict=True)
        link = args.work/name
        if not link.exists():
            link.symlink_to(target)
        if target != link.resolve():
            p.error("Coverage input already refers to a different acquisition.")
    outputs = ["nsc_object.csv", "validation/gaia.csv", "detections.csv", "external_labels.csv"]
    with run_stage(args.work, args.config, args.region, "pilot-inputs-v05",
                   inputs=inputs, outputs=outputs) as go:
        if not go:
            return
        cache = access.Cache(args.work/"cache")
        index = pd.read_csv(args.work/"nsc_object_index.csv", dtype={"objectid":str})
        if index.objectid.duplicated().any():
            raise ValueError("NSC archive index has duplicate object aliases.")
        measurements = access.fetch_measurements_for_objects(cache, index.objectid.sort_values())
        measurements.to_csv(args.work/"nsc_object.csv", index=False)
        print(f"{len(measurements):,} released NSC measurements acquired", flush=True)
        columns = ("source_id,ra,dec,ref_epoch,pmra,pmdec,pmra_error,pmdec_error,parallax,"
                   "ruwe,phot_g_mean_mag,phot_rp_mean_mag,parallax_error,radial_velocity,astrometric_params_solved")
        radius = field["radius"] + settings["calibration_buffer_deg"] + .004
        gaia = access.checked_spatial(cache, "nsc", columns, "gaia_dr3.gaia_source",
                                      access.q3c(*center,radius), center[1]-radius-1e-8,
                                      center[1]+radius+1e-8)
        (args.work/"validation").mkdir(exist_ok=True)
        gaia.to_csv(args.work/"validation"/"gaia.csv", index=False)
        standardize(args.work,{},center,field["radius"],"object")


if __name__ == "__main__":
    main()
