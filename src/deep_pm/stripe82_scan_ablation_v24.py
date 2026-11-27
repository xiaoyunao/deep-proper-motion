"""Fixed HLC stars: retain all NSC members and compare first versus all SDSS runs."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .astrometry import project,years
from .continuous_reference import read_offsets,verify_directory
from .coordinate_metric import fit_vector_motion,native_covariance,projected_covariance,reference_covariance
from .standard_motion import icrs_motion
from .stream_json import array_items
from .io import digest,save_json


def rank(value):return sha256(("v24-hlc-run-before-fit:"+str(value)).encode()).hexdigest()


def select(frame):
    eligible=frame.loc[frame.in_nominal & frame.time_opportunity & frame.sdss_runs.ge(2)
                       & frame.xiao_e.notna() & frame.nsc_e.notna() & frame.r.between(21,23,inclusive="left")].copy()
    eligible["faint"]=eligible.r>=22
    eligible["west"]=eligible.ra>=180
    parts=[]
    for _,pool in eligible.groupby("west"):
        counts=pool.groupby(["origin_core","faint"]).size().unstack(fill_value=0).reindex(columns=[False,True],fill_value=0)
        supported=counts.loc[counts.min(axis=1)>=6]
        for core in sorted(supported.index,key=rank)[:2]:
            for _,group in pool[pool.origin_core==core].groupby("faint"):
                ordered=group.identity.map(rank).sort_values().index[:24]
                parts.append(pool.loc[ordered,["identity","r","origin_core","origin_slot","sdss_runs"]])
    if not parts:raise ValueError("No common repeated-scan stars with supported magnitude strata")
    return pd.concat(parts,ignore_index=True)


def fit(data,row,mask,solution):
    frame=data.loc[mask]
    center=row["projection_center_deg"]
    t=years(frame.observation_time.to_numpy())
    groups=np.asarray(row["physical_groups"])[mask]
    xy=project(frame.ra_deg,frame.dec_deg,center)-np.array([solution[g][0] for g in groups])
    sigma=np.hypot(frame[["sigma_east_mas","sigma_north_mas"]].to_numpy(),50.)
    noise=projected_covariance(frame.ra_deg,frame.dec_deg,center,native_covariance(sigma))
    model=fit_vector_motion(xy,t,noise,2015.5)
    common,_=reference_covariance(model.influence,np.stack([solution[g][1] for g in groups],axis=1))
    result=icrs_motion(model.coeff[0],model.coeff[1],center,model.cov+common)
    rr=model.residual
    z=np.sqrt(np.einsum("ni,nij,nj->n",rr,np.linalg.inv(noise),rr))
    passed=np.max(result["pm_error_masyr"])<=10 and np.quantile(z,.68)<=3 and np.ptp(t)>=5
    return np.asarray(result["pm_icrs_masyr"]),np.asarray(result["pm_error_masyr"]),passed


def segment(job):
    root,entry,targets=job;source=Path(entry["source"])
    folder=root/"sources"/entry["core"]
    file=folder/"private_standard_motion.json.gz";receipt=json.loads((folder/"v22_source_core_summary.json").read_text())
    if digest(file)!=receipt["private_output_sha256"][file.name]:raise ValueError("Source receipt mismatch")
    by_slot={int(r.origin_slot):r for r in targets.itertuples(index=False)}
    rows={i:r for i,r in enumerate(array_items(file)) if i in by_slot}
    if len(rows)!=len(by_slot):raise ValueError("Missing selected member history")
    verify_directory(source,("detections.csv",))
    needed={k for row in rows.values() for k in row["detection_ids"]}
    cols=["detection_id","ra_deg","dec_deg","survey","exposure_id","observation_time","sigma_east_mas","sigma_north_mas"]
    frame=pd.concat([p.loc[p.detection_id.isin(needed)] for p in pd.read_csv(source/"detections.csv",usecols=cols,chunksize=200000)])
    if frame.detection_id.duplicated().any() or len(frame)!=len(needed):raise ValueError("Physical members did not reproduce")
    frame=frame.set_index("detection_id");graphs={};answers=[]
    for slot,row in rows.items():
        data=frame.loc[row["detection_ids"]];model_id=row["reference_component"]
        if model_id not in graphs:
            path=root/"reference"/f"component-{int(model_id):04d}"/"private_c0_unfiltered.npz"
            graphs[model_id]=(digest(path),read_offsets(path))
        hash_,solution=graphs[model_id]
        if hash_!=row["reference_model_sha256"]:raise ValueError("Incompatible shared calibration")
        sdss=data.survey.eq("SDSS");runs=data.loc[sdss].groupby("exposure_id").observation_time.min().sort_index()
        item=by_slot[slot]
        if len(runs)!=item.sdss_runs or len(runs)<2:raise ValueError("Unexpected physical scan count")
        full,se,ok=fit(data,row,np.ones(len(data),bool),solution)
        np.testing.assert_allclose(full,row["pm_icrs_masyr"],rtol=2e-7,atol=2e-5)
        np.testing.assert_allclose(se,row["pm_error_masyr"],rtol=2e-7,atol=2e-6)
        if not ok:raise ValueError("Original frozen scientific cuts failed")
        mask=(~sdss | data.exposure_id.eq(runs.idxmin())).to_numpy()
        earliest,se_one,passed=fit(data,row,mask,solution)
        answers.append(dict(identity=item.identity,r=item.r,runs=len(runs),nsc=int((~sdss).sum()),first_run=runs.idxmin(),
                            all_e=full[0],all_n=full[1],first_e=earliest[0],first_n=earliest[1],first_pass=bool(passed),
                            se_all=float(np.linalg.norm(se)),se_first=float(np.linalg.norm(se_one))))
    return answers


def run(root,directory,workers):
    path=directory/"private_pairs.parquet";summary=json.loads((directory/"v24_stripe82_comparison_summary.json").read_text())
    if digest(path)!=summary["pair_sha256"]:raise ValueError("Changed fixed HLC identity")
    spec=select(pd.read_parquet(path,columns=["identity","ra","r","in_nominal","time_opportunity","sdss_runs","origin_core","origin_slot","xiao_e","nsc_e"]))
    selection=directory/"private_run_ablation_selection.json"
    ticket=dict(pairs=summary["pair_sha256"],identity=spec.identity.astype(int).tolist(),implementation=digest(Path(__file__)))
    if selection.exists() and json.loads(selection.read_text())!=ticket:raise ValueError("Changed fixed ablation selection")
    save_json(selection,ticket)
    plan=json.loads((root/"registration/private_full_coordinate_plan.json").read_text())
    entries={e["core"]:e for e in plan["cores"]}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        parts=list(pool.map(segment,[(root,entries[name],g) for name,g in spec.groupby("origin_core")]))
    results=pd.DataFrame([row for p in parts for row in p]).merge(pd.read_parquet(path,columns=["identity","pm_e","pm_n"]),on="identity",validate="one_to_one")
    if len(results)!=len(spec):raise ValueError("Unaccounted ablation star")
    detail=directory/"private_run_ablation_results.parquet";results.to_parquet(detail,index=False)
    out={}
    for low in (21,22):
        group=results[results.r.between(low,low+1,inclusive="left")]
        for label,frame in (("all_fixed_members",group),("both_pass",group.loc[group.first_pass])):
            stats={}
            if len(frame):
                for model in ("all","first"):
                    delta=frame[[model+"_e",model+"_n"]].to_numpy()-frame[["pm_e","pm_n"]].to_numpy()
                    med=np.median(delta,axis=0);rad=np.linalg.norm(delta,axis=1)
                    stats[model]=dict(median=med,nmad=1.4826*np.median(abs(delta-med),axis=0),p68=np.quantile(rad,.68),
                                      gt20=int((rad>20).sum()),median_vector_SE=frame["se_"+model].median())
            out[f"{low}-{low+1}_{label}"]=dict(n=len(frame),methods=stats)
    save_json(directory/"v24_stripe82_scan_ablation_summary.json",dict(scope=summary["scope"],selection=digest(selection),
              rows=len(results),cores=spec.origin_core.nunique(),accepted_first_run=int(results.first_pass.sum()),by_magnitude=out,
              private_result_sha256=digest(detail),meaning="Fixed four-way HLC stellar identities and late detections, original C0/W0. "
              "Earliest physical SDSS run by observed MJD. Conditional measurement experiment, not single-run discovery completeness."))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("root",type=Path);p.add_argument("directory",type=Path);p.add_argument("--workers",type=int,default=2)
    a=p.parse_args()
    if not os.getenv("SLURM_JOB_ID") or not 1<=a.workers<=int(os.environ["SLURM_CPUS_PER_TASK"]):p.error("Use assigned Slurm CPUs")
    run(a.root,a.directory,a.workers)
