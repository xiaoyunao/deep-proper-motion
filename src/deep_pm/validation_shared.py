"""Fit one common graph across two real northern processing cores."""
import argparse
import json
import os
from pathlib import Path

import healpy as hp
import numpy as np
import pandas as pd

from .__main__ import tomllib
from .astrometry import project
from .exposure_catalogues import inside
from .exposure_merge import FILES, merge
from .exposure_model import hierarchy, observation_keys
from .exposure_pilot import held_galaxies
from .io import save_json
from .pilot_production import quality_detections
from .precision_v03 import calibration
from .provenance import run_stage
from .validation_audit import graph_support


def compute(work, fields, calibration_settings):
    config=tomllib.loads(Path("configs/precision-v03.toml").read_text())
    settings=tomllib.loads(Path("configs/spatial-v08.toml").read_text())
    settings["bootstrap_draws"]=400
    center=(350.,0.)
    data,xy,t,sigma,_=quality_detections(work,center,config,settings["production"])
    nsc=data.survey.eq("NSC").to_numpy()
    veto=pd.read_csv(work/"spectra_veto.csv")
    c0,_,refs=calibration(work,center,config,project(veto.ra,veto.dec,center),
                          return_references=True,arrays=(data,xy,t,sigma))
    parent,fine=observation_keys(data)
    key,c1,_=hierarchy(data,refs,xy,sigma,config,calibration_settings,c0)
    support=np.isin(key,list(c1))
    keys=set(c1)
    per_core={}
    supported_groups=[]
    held_sets=[]
    pix=hp.ang2pix(128,data.ra_deg,data.dec_deg,lonlat=True)
    for name,spec in fields.items():
        core=inside(data.ra_deg,data.dec_deg,spec)
        supported_groups.append(set(key[core & support]))
        held=refs.loc[refs.is_holdout & core[refs.index]]
        held_sets.append(set(held.reference))
        stats=held_galaxies(held,data,xy,nsc,(parent,key),(c0,c1),pix,fine,settings)["comparison"]
        per_core[name]=dict(astrometric_measurements=int(core.sum()),
             shared_solution_coverage=int((core & support).sum()),held_galaxies=stats)
    source_totals=json.loads((work/"acquisition_summary.json").read_text())["counts"]["detections.csv"]
    shared=set.intersection(*supported_groups)
    save_json(work/"shared_summary.json",dict(cores=per_core,
        graph=graph_support(refs,key,c1),deduplicated_measurements=source_totals,
        supported_groups_used_in_both=len(shared),
        shared_sdss_groups=sum(k.startswith("SDSS:") for k in shared),
        shared_nsc_groups=sum(k.startswith("NSC:") for k in shared),
        holdout_references_seen_in_both=len(set.intersection(*held_sets)),
        calibrated_groups=len(keys)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("work","parts","config"):
        p.add_argument(key,type=Path)
    a=p.parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        p.error("Use a Slurm compute node")
    a.work.mkdir(parents=True,exist_ok=True)
    settings=tomllib.loads(a.config.read_text())
    fields={name:settings["fields"][name] for name in ("sparse_north_inner","sparse_north_outer")}
    parts=[a.parts/name for name in fields]
    input_names=[]
    for name,part in zip(fields,parts):
        manifest=a.work/f"{name}_input.json"
        if not manifest.exists():
            manifest.symlink_to((part/"run_manifest.json").resolve(strict=True))
        input_names.append(manifest.name)
    with run_stage(a.work,a.config,"north_shared","validation-shared-input-v10",inputs=input_names,
                   outputs=(*FILES,"acquisition_summary.json")) as go:
        if go:
            merge(a.work,parts,dict(ra=[349.,353.],dec=[1.,5.]))
    with run_stage(a.work,a.config,"north_shared","validation-shared-v10",
                   inputs=input_names,outputs=["shared_summary.json"]) as go:
        if go:
            compute(a.work,fields,settings["calibration"])


if __name__=="__main__":
    main()
