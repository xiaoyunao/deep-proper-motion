"""Compare a full signed native pilot to a fresh counted TAP acquisition on the same field."""
import argparse
import json
import os
from pathlib import Path

import pandas as pd

from .exposure_inputs import acquire
from .footprint_intake_v24 import OUTPUTS, SETTINGS
from .full_reference_input import extract_one
from .io import digest, save_json
from .provenance import run_stage


def run(base,catalogues,sas):
    work=base/"v2.4"
    pilot_rows=[]
    for f in (work/"registration").glob("v24_intake_*_summary.json"):
        if f.name.split("_")[2].isdigit():
            pilot_rows.extend(r for r in json.loads(f.read_text())["rows"] if r.get("measurements",0)>0)
    chosen=min(pilot_rows,key=lambda r:(r["rows"],r["core"]))["core"]
    plan=work/"registration/private_full_footprint_plan.json"
    spec=json.loads(plan.read_text())["new"][chosen]
    pointer=work/"resolved"/(chosen+".json")
    binding=json.loads(pointer.read_text()) if pointer.is_file() else {}
    old=Path(binding.get("source",work/"intake"/chosen))
    original_ref=Path(binding.get("shard",work/"input"/chosen))/"reference.parquet"
    check=work/"transport-qualification"/os.environ["SLURM_JOB_ID"]
    fresh=check/"intake"/chosen
    fresh.mkdir(parents=True,exist_ok=True)
    config=fresh/"registered.toml"
    config.write_text((old/"registered.toml").read_text()+'\n# fresh independent TAP query cache\n')
    with run_stage(fresh,config,chosen,"native-to-counted-tap",outputs=OUTPUTS) as go:
        if go:
            acquire(fresh,spec,SETTINGS,catalogues,sas,1)
    ref=extract_one((dict(core=chosen,source=str(fresh),source_receipt_sha256=digest(fresh/"run_manifest.json"),
                          center=[sum(spec[k])/2 for k in ("ra","dec")]),check/"reference"))
    science={}
    keys={"detections.csv":["detection_id"],"external_labels.csv":["detection_id"],
          "nsc_object.csv":["measid"],"official_nsc.csv":["id"],"chips.csv":["exposure","ccdnum"],
          "validation/gaia.csv":["source_id"],"sdss.csv":None}
    for name,identity in keys.items():
        f1,f2=old/name,fresh/name
        same=digest(f1)==digest(f2)
        x=pd.read_csv(f1,low_memory=False)
        if not same:
            y=pd.read_csv(f2,low_memory=False)
            if identity:
                if x.duplicated(identity).any() or y.duplicated(identity).any():
                    raise ValueError(f"Nonunique validation identities in {name}")
                x=x.sort_values(identity).reset_index(drop=True)
                y=y.sort_values(identity).reset_index(drop=True)
            pd.testing.assert_frame_equal(x,y,check_exact=True)
        science[name]=dict(rows=len(x),byte_identical=same,all_fields_identical=True)
    new_ref=check/"reference"/chosen/"reference.parquet"
    x=pd.read_parquet(original_ref).sort_values("detection_id").reset_index(drop=True)
    y=pd.read_parquet(new_ref).sort_values("detection_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(x,y,check_exact=True)
    save_json(work/"registration/v24_transport_qualification_summary.json",
              dict(registered_footprint=digest(plan),pilot=chosen,science_tables=science,
                   reference_measurements=ref["measurements"],reference_all_fields_identical=True,
                   reference_parquet_identical=digest(original_ref)==digest(new_ref),
                   selection="Minimum signed positive-reference pilot detection count. Independent empty TAP cache; separate COUNT before bounded data retrieval.",
                   slurm_job=os.environ["SLURM_JOB_ID"]))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("base","catalogues","sas"):
        p.add_argument(k,type=Path)
    a=p.parse_args()
    if not os.getenv("SLURM_JOB_ID") or os.environ.get("DPM_NSC_TRANSPORT")!="counted-tap":
        p.error("Use a Slurm allocation and the explicitly selected counted TAP transport")
    run(a.base,a.catalogues,a.sas)
