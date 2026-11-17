"""Retain per-SDSS-field-pair overlap and covariance evidence in Scratch."""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .__main__ import tomllib
from .astrometry import project
from .exposure_model import hierarchy, observation_keys
from .io import save_json
from .overlap_calibration import overlap_trial, pair_audit
from .pilot_production import quality_detections
from .precision_v03 import calibration
from .provenance import run_stage
from .validation_audit import sdss_overlap_rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "inputs", "sas", "config"):
        p.add_argument(name, type=Path)
    p.add_argument("region")
    args = p.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        p.error("Use a Slurm compute node.")
    work = args.output/args.region
    work.mkdir(parents=True, exist_ok=True)
    receipt = work/"inputs.json"
    if not receipt.exists():
        receipt.symlink_to((args.inputs/"run_manifest.json").resolve(strict=True))
    if receipt.resolve() != (args.inputs/"run_manifest.json").resolve(strict=True):
        raise ValueError("Frozen input changed")
    config = tomllib.loads(args.config.read_text())
    field = config["fields"][args.region]
    center = tuple(np.mean(field[k]) for k in ("ra", "dec"))
    fit = tomllib.loads(Path("configs/precision-v03.toml").read_text())
    prod = tomllib.loads(Path("configs/spatial-v08.toml").read_text())["production"]
    with run_stage(work, args.config, args.region, "overlap-pairs-v11",
                   inputs=("inputs.json",), outputs=("private_field_pair_audit.json",)) as go:
        if not go:
            return
        data, xy, time, sigma, _ = quality_detections(args.inputs, center, fit, prod)
        veto = pd.read_csv(args.inputs/"spectra_veto.csv")
        c0, _, refs = calibration(args.inputs, center, fit, project(veto.ra, veto.dec, center),
                                  return_references=True, arrays=(data, xy, time, sigma))
        fine, c1, _ = hierarchy(data, refs, xy, sigma, fit, config["calibration"], c0)
        keys = observation_keys(data)[0], fine
        models = c0, c1
        train, counts = overlap_trial(args.sas, refs, data, xy, sigma, keys, models, center, config["overlap"])
        held, hold_counts = sdss_overlap_rows(args.sas, refs, data, xy, keys, models, center, holdout=True)
        _, c1b, _ = hierarchy(data, refs, xy, sigma, fit, config["calibration"], c0, overlaps=train)
        save_json(work/"private_field_pair_audit.json",
                  dict(training=counts, holdout=hold_counts, field_pairs=pair_audit(train, held, (*models, c1b))))


if __name__ == "__main__":
    main()
