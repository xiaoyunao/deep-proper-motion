"""Audit actual CCD and SDSS field overlaps in the existing pilot data."""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .__main__ import tomllib
from .astrometry import project
from .exposure_model import hierarchy, observation_keys
from .io import save_json
from .pilot_production import quality_detections
from .precision_v03 import calibration
from .provenance import run_stage
from .validation_audit import held_conditions, sdss_field_overlap


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("root",type=Path)
    p.add_argument("name")
    p.add_argument("inputs",type=Path)
    p.add_argument("sas",type=Path)
    p.add_argument("config",type=Path)
    a=p.parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        p.error("Use a Slurm compute node")
    work=a.root/a.name
    work.mkdir(parents=True,exist_ok=True)
    cfg=tomllib.loads(a.config.read_text())
    spec=cfg["fields"][a.name]
    center=tuple(np.mean(spec[k]) for k in ("ra","dec"))
    fit=tomllib.loads(Path("configs/precision-v03.toml").read_text())
    production=tomllib.loads(Path("configs/spatial-v08.toml").read_text())["production"]
    source=(a.inputs/a.name).resolve(strict=True)
    link=work/"input_manifest.json"
    if not link.exists():
        link.symlink_to(source/"run_manifest.json")
    with run_stage(work,a.config,a.name,"validation-existing-v10",inputs=[link.name],
                   outputs=["boundary_summary.json"]) as go:
        if not go:
            return
        data,xy,time,sigma,_=quality_detections(source,center,fit,production)
        qso=pd.read_csv(source/"spectra_veto.csv")
        c0,_,refs=calibration(source,center,fit,project(qso.ra,qso.dec,center),return_references=True,
                              arrays=(data,xy,time,sigma))
        key,c1,_=hierarchy(data,refs,xy,sigma,fit,cfg["calibration"],c0)
        parent,_=observation_keys(data)
        edges=sdss_field_overlap(a.sas,refs,data,xy,(parent,key),(c0,c1),center,spec)
        conditions=held_conditions(refs,data,xy,key,c1)
        save_json(work/"boundary_summary.json",dict(sdss_adjacent_field=edges,held_conditions=conditions))


if __name__=="__main__":
    main()
