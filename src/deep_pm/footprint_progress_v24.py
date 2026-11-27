"""A batch-area ledger from signed completed acquisitions; never infer sky from sources."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .continuous_summary import area
from .io import digest,save_json


def read(path):
    return json.loads(path.read_text())


def forecast(root,cells,completed):
    load_file=root/"capacity/private_unit_degree_load.parquet"
    proof=read(root/"capacity/v24_detector_load_summary.json")
    if digest(load_file)!=proof["load_sha256"]:raise ValueError("Changed physical detector load map")
    load=pd.read_parquet(load_file)
    sky={}
    for row in load.itertuples():
        if -90<=row.d<90 and -360<=row.a<720:
            key=(int(row.a)%360,int(row.d))
            sky[key]=sky.get(key,0)+int(row.meas)
    estimates=[]
    for name,spec in cells.items():
        west,east=spec["ra"];south,north=spec["dec"]
        w,e=west-.06,east+.06;s,n=max(-90,south-.06),min(90,north+.06)
        mass=0.
        for j in range(int(np.floor(s)),int(np.ceil(n))):
            latitude=(np.sin(np.deg2rad(min(n,j+1)))-np.sin(np.deg2rad(max(s,j))))/(np.sin(np.deg2rad(j+1))-np.sin(np.deg2rad(j)))
            for i in range(int(np.floor(w)),int(np.ceil(e))):
                mass+=latitude*(min(e,i+1)-max(w,i))*sky.get((i%360,j),0.)
        stripe=abs((south+north)/2)<1.6 and ((west+east)/2<53 or (west+east)/2>=307)
        row=completed.get(name,{})
        estimates.append(dict(stripe=stripe,load_bin=int(np.searchsorted([50000,200000,600000,2000000],mass)),
                         scale=1+mass/200000.,signed=name in completed,wall=row.get("wall_s"),size=row.get("raw_bytes"),
                         native_rows=row.get("rows"),reference_rows=row.get("measurements")))
    df=pd.DataFrame(estimates);done=df[df.signed];total=np.zeros((4,3));unmeasured=0;fallback=0;strata=[]
    for (stripe,b),group in df[~df.signed].groupby(["stripe","load_bin"]):
        sample=done[(done.stripe==stripe)&(done.load_bin==b)]
        native=len(sample)
        if native<3:
            sample=done[done.load_bin==b]
            fallback+=len(group)
        if len(sample)<3:
            unmeasured+=len(group);continue
        scale=group.scale.sum()
        for k,metric in enumerate(("wall","size","native_rows","reference_rows")):
            total[k]+=scale*np.quantile(sample[metric]/sample.scale,[.25,.5,.75])
        strata.append(dict(stripe=bool(stripe),load_bin=int(b),remaining=len(group),same_stratum_receipts=native,
                           receipts_used=len(sample)))
    return dict(remaining=len(df)-len(done),without_minimum_calibrators=unmeasured,
                using_pooled_latitude_or_without_calibrators=fallback,strata=strata,
                supported_remaining_worker_hours=total[0]/3600,
                supported_remaining_intake_node_hours_at_2p2_billed_threads_per_worker=total[0]*2.2/(256*3600),
                supported_remaining_input_TiB=total[1]/2**40,
                supported_remaining_halo_detection_rows=total[2],supported_remaining_halo_reference_rows=total[3],
                scope="Intake only. Three descriptive forecasts use P25/P50/P75 completed worker wall/byte rates "
                      "within observed CCD-load and Stripe strata. Scale 1+nmeas/200000 from the halo rectangle. "
                      "2.2 is the actual shared reservation for a regular 10-worker/40GiB job; not an invoice. "
                      "Live dense stragglers can exceed these rates. Output, graph, retries, and missing strata are additional.")

def run(base,out):
    root=base/"v2.4";reg=root/"registration"
    manifest=reg/"private_full_footprint_plan.json"
    sha=digest(manifest);cells=read(manifest)["new"]
    completed={}
    for p in sorted(reg.glob("v24_intake_*_summary.json")):
        if not p.name[len("v24_intake_"):-len("_summary.json")].isdigit():continue
        pilot=read(p)
        if pilot["footprint_sha256"]!=sha:raise ValueError("Changed pilot footprint")
        completed.update({r["core"]:r for r in pilot["rows"] if "error" not in r})
    failures=set()
    for p in (root/"batch-receipts").glob("*.json"):
        if p.name.endswith(".failed.json"):
            failures.add(p.name.split(".")[0]);continue
        rec=read(p)
        if rec["manifest_sha256"]!=sha or rec["core"] not in cells:raise ValueError("Unregistered intake result")
        completed[rec["core"]]=rec
    failures-=set(completed)
    if not completed:raise ValueError("No completed input receipts")
    values=list(completed.values());r=np.array([row["rows"] for row in values])
    result=dict(signed_footprint=sha,registered_new_core_count=len(cells),completed_new_inputs=len(values),
                remaining_new_inputs=len(cells)-len(values),failed_unrepaired_inputs=len(failures),
                input_rectangles_nominal_sqdeg=sum(area(cells[k]) for k in completed),
                time_opportunity_in_completed_new_inputs_sqdeg=sum(cells[k]["time_area"] for k in completed),
                input_rows_including_halo=int(r.sum()),input_bytes=sum(v["raw_bytes"] for v in values),
                reference_rows_including_halo=sum(v["measurements"] for v in values),
                cores_with_local_reference_rows=sum(v["measurements"]>0 for v in values),
                core_rows_p10_p50_p90=np.quantile(r,[.1,.5,.9]),
                worker_seconds_sum=sum(v["wall_s"] for v in values),
                peak_individual_rss_mib=max(v["rss_mib"] for v in values),
                slurm_job=os.environ["SLURM_JOB_ID"],
                interpretation="Completed intake with signed source and reference extraction. Current graph/fit results are accounted separately. Shared detections across core halos occur in these row counts.")
    result["remaining_intake_planning"]=forecast(root,cells,completed)
    save_json(out,result)
    print("INPUTS",result["completed_new_inputs"],"ROWS",int(r.sum()),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("base",type=Path);p.add_argument("output",type=Path);a=p.parse_args()
    if not os.getenv("SLURM_JOB_ID"):p.error("Use a Slurm node")
    run(a.base,a.output)
