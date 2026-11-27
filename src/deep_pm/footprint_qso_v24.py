"""Post-closure spectral evaluation on the published positions and real blind catalogue."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import json
import os
from pathlib import Path

from astropy.io import fits
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .exposure_catalogues import coordinates,inside
from .io import digest,save_json
from .stream_json import array_items

_TREE=None
PM=["pm_e","pm_n","se_e","se_n"]
RAD=180*3600/np.pi


def candidates(data,limit,reverse=False,workers=1):
    # Neighbours beyond gate+margin cannot change either uniqueness test.
    # Reverse (QSO -> published target) requires all neighbours, including
    # a third spectrum close to a target. Keep only its small candidate pool.
    sky=coordinates(data.ra.to_numpy()%360,data.dec.to_numpy())
    good=np.isfinite(sky).all(axis=1)&data.dec.between(-90,90).to_numpy()
    frame=data.loc[good].reset_index(drop=True);sky=sky[good]
    bound=2*np.sin((limit+.3)/(2*RAD))
    dist,near=_TREE.query(sky,k=2,distance_upper_bound=bound,workers=workers)
    if reverse:
        ix=np.flatnonzero(np.isfinite(dist[:,0]))
        lists=_TREE.query_ball_point(sky[ix],bound,workers=workers)
        q=np.concatenate([np.asarray(v,dtype=int) for v in lists]) if len(ix) else np.empty(0,int)
        j=np.repeat(ix,np.asarray([len(v) for v in lists],dtype=int))
        separation=2*np.arcsin(np.minimum(1,np.linalg.norm(sky[j]-_TREE.data[q],axis=1)/2))*RAD
    else:
        dist=2*np.arcsin(np.minimum(1,dist/2))*RAD
        j=np.flatnonzero((dist[:,0]<=limit)&(dist[:,1]-dist[:,0]>.3))
        q=near[j,0];separation=dist[j,0]
    out=frame.iloc[j].copy()
    out["q"]=q;out["separation"]=separation
    return out


def unique(cand,reverse):
    if not len(cand):return cand
    if not reverse:
        return cand.loc[~cand.q.duplicated(keep=False)&cand.valid].copy()
    # Overlapping acquisition haloes may contain one physical object many times.
    key=["q","external"]
    duplicate=cand.loc[cand.duplicated(key,keep=False)]
    if len(duplicate) and (duplicate.groupby(key)[["ra","dec","valid",*PM]].nunique(dropna=False)>1).any().any():
        raise ValueError("Conflicting repeated published catalogue state")
    cand=cand.drop_duplicates(key).sort_values(["q","separation","external"])
    order=cand.groupby("q").cumcount()
    first=cand.loc[order==0].copy()
    second=cand.loc[order==1].set_index("q").separation
    gap=first.q.map(second).fillna(np.inf)-first.separation
    first=first.loc[(first.separation<=1)&(gap>.3)]
    return first.loc[~first.external.duplicated(keep=False)&first.valid].copy()


def valid(frame):
    x=frame[PM].to_numpy()
    return np.isfinite(x).all(axis=1)&(np.max(abs(x[:,:2]),axis=1)<1e5)&(np.min(x[:,2:],axis=1)>0)


def owns_spectrum(cat,entries):
    # Bound work to the one-degree cells intersected by the registered boxes.
    a,d=cat.ra.to_numpy()%360,cat.dec.to_numpy()
    keys=np.floor(a).astype(int)+360*np.clip(np.floor(d+90).astype(int),0,179)
    blocks=pd.Series(np.arange(len(cat))).groupby(keys).indices
    mask=np.zeros(len(cat),bool)
    for entry in entries:
        spec=entry["spec"]
        w,e=spec["ra"];south,north=spec["dec"]
        for x in range(int(np.floor(w)),int(np.ceil(e))):
            for y in range(max(0,int(np.floor(south+90))),min(180,int(np.ceil(north+90)))):
                ix=blocks.get(x%360+360*y)
                if ix is None:continue
                hit=ix[inside(a[ix],d[ix],spec)]
                if np.any(mask[hit]):raise ValueError("Spectroscopic control overlaps two owners")
                mask[hit]=True
    return mask


def per_core(task):
    entry,root,cache,plan_sha,survey_sha=task
    label=entry["core"];source=Path(entry["source"]);folder=cache/"inputs"/label
    folder.mkdir(exist_ok=True,parents=True)
    files=[folder/(k+".parquet") for k in ("nsc","r")]
    proof=folder/"receipt.json"
    source_receipt=source/"run_manifest.json"
    ticket=dict(plan=plan_sha,spectra=survey_sha,input_sha256=digest(source_receipt),implementation=digest(Path(__file__)))
    if ticket["input_sha256"]!=entry["source_receipt_sha256"]:raise ValueError("Changed acquisition before validation")
    if proof.exists():
        r=json.loads(proof.read_text())
        if ticket!=r["ticket"] or any(digest(p)!=r["files"][p.name] for p in files):raise ValueError("Changed post-freeze spectral cache")
        return
    data=pd.read_csv(source/"official_nsc.csv",usecols=["id","ra","dec","pmra","pmdec","pmraerr","pmdecerr","deltamjd"],
                     dtype={"id":str}).rename(columns=dict(zip(["id","pmra","pmdec","pmraerr","pmdecerr"],["external",*PM])))
    data["valid"]=valid(data)&data.deltamjd.gt(0)
    candidates(data,1.,True).to_parquet(files[0],index=False)
    rows=[]
    if entry.get("save_r_comparison"):
        f=root/"sources"/label/"private_r_control.json.gz"
        s=json.loads((f.parent/"v22_source_core_summary.json").read_text())
        if digest(f)!=s["private_output_sha256"][f.name]:raise ValueError("Changed physical r-control member set")
        for r in array_items(f):
            if r["control_kind"]=="physical":
                ra,dec=r["position_deg"]
                ident=sha256("\0".join(sorted(r["detection_ids"])).encode()).hexdigest()
                rows.append([ident,ra,dec,*r["pm_icrs_masyr"],*r["pm_error_masyr"]])
    data=pd.DataFrame(rows,columns=["external","ra","dec",*PM]).astype({n:float for n in ["ra","dec",*PM]})
    data["valid"]=True
    candidates(data,.5).to_parquet(files[1],index=False)
    save_json(proof,dict(ticket=ticket,files={p.name:digest(p) for p in files}))


def run(root,spectra,xiao,output,workers):
    global _TREE
    import duckdb
    from multiprocessing import get_context
    plan=root/"registration/private_full_coordinate_plan.json"
    entries=json.loads(plan.read_text())["cores"]
    closure=root/"catalogue/v22_catalogue_ownership_summary.json"
    owner=json.loads(closure.read_text());binding=root/"catalogue/private_closure_binding.json"
    if digest(binding)!=owner["private_input_binding_sha256"] or json.loads(binding.read_text())["plan_sha256"]!=digest(plan):
        raise ValueError("Spectral labels require a closed source catalogue")
    output.mkdir(exist_ok=True,parents=True)
    cat=pd.read_csv(spectra,usecols=["ra","dec","reference_r","origin"])
    _TREE=cKDTree(coordinates(cat.ra,cat.dec))
    key=digest(plan);checksum=digest(spectra)
    ticket=dict(closure=digest(closure),registered=key,spectra=checksum,xiao_stat=[xiao.stat().st_size,xiao.stat().st_mtime_ns],
                implementation=digest(Path(__file__)),workers_scientifically_interchangeable=True)
    proof=output/"private_input_binding.json"
    if proof.exists() and json.loads(proof.read_text())!=ticket:raise ValueError("Changed scientific QSO validation inputs")
    save_json(proof,ticket)
    final=output/"private_final_spectral_pairing.parquet"
    receipt=output/"v24_spectral_pairing_summary.json"
    if receipt.exists():
        if json.loads(receipt.read_text())["private_pair_sha256"]!=digest(final):raise ValueError("Changed spectral pairing")
        return
    with ProcessPoolExecutor(max_workers=workers,mp_context=get_context("fork")) as pool:
        for _ in pool.map(per_core,[(e,root,output,key,checksum) for e in entries],chunksize=4):pass
    db=duckdb.connect(str(root/"catalogue/catalogue.duckdb"),read_only=True)
    db.execute(f"SET threads={workers}")
    db.execute("SET memory_limit='4GB'")
    stream=db.execute("""SELECT src AS external,ra_deg AS ra,dec_deg AS dec,pm_e,pm_n,se_e,se_n,
         core,slot,kind,pixel128,sdss_runs,sdss_years,nsc_years,nsc_span,joint_span,nsc_nights,nsc_seasons,instruments,bands,
         maximum_group_leverage,pm_cov_en,independent_pm_variance_e,independent_pm_variance_n,
         reference_pm_variance_e,reference_pm_variance_n,component,TRUE AS valid FROM accepted""").fetch_record_batch(150000)
    parts=[];n=0
    for batch in stream:
        df=batch.to_pandas();n+=len(df)
        part=candidates(df,.5,workers=workers)
        if len(part):parts.append(part)
    db.close()
    if n!=owner["retained_common_epoch_solutions"]:raise ValueError("Source catalogue count changed")
    ours=unique(pd.concat(parts,ignore_index=True),False)
    # Query-level identity is the frozen order of the union of spectroscopy.
    cat["q"]=np.arange(len(cat))
    cat["in_nominal"]=owns_spectrum(cat,entries)
    merged=cat.merge(ours.drop(columns=["ra","dec","valid"]).rename(columns={c:"ours_"+c for c in ["external","separation",*PM]}),
                     on="q",how="left",validate="one_to_one")
    parts=[]
    with fits.open(xiao,memmap=True,character_as_bytes=True) as f:
        table=f[1].data
        for start in range(0,len(table),300000):
            v=table[start:start+300000]
            good=np.char.strip(v["PM_SOURCE"].astype("U8"))=="DESI"
            if not np.any(good):continue
            data=pd.DataFrame(dict(external=start+np.flatnonzero(good),ra=v["RA"][good].astype(float),
                    dec=v["DEC"][good].astype(float),**{col:v[name][good].astype(float) for col,name in zip(PM,["PMRA","PMDEC","PMRA_ERR","PMDEC_ERR"])}))
            data["valid"]=valid(data);part=candidates(data,1.,True,workers)
            if len(part):parts.append(part)
    for method in ("xiao","nsc","r"):
        if method!="xiao":
            combined=pd.concat([pd.read_parquet(output/"inputs"/entry["core"]/(method+".parquet")) for entry in entries],ignore_index=True)
        else:combined=pd.concat(parts,ignore_index=True)
        if method=="r":
            connected=combined.drop_duplicates(["q","external"])
            if len(connected)!=len(combined) and (combined.groupby(["q","external"])[PM].nunique()>1).any().any():
                raise ValueError("Competing r-control halo copies")
            combined=connected
        accepted=unique(combined,method!="r")
        columns=["q","external",*PM]
        merged=merged.merge(accepted[columns].rename(columns={c:method+"_"+c for c in columns[1:]}),on="q",how="left",validate="one_to_one")
    if list(ticket["xiao_stat"])!=[xiao.stat().st_size,xiao.stat().st_mtime_ns]:raise ValueError("Xiao catalogue changed")
    merged=merged[merged.in_nominal].copy()
    merged.to_parquet(final,index=False,compression="zstd")
    save_json(receipt,dict(catalogue=n,spectroscopic_archive_rows=len(cat),nominal_spectroscopic_parent=len(merged),matched={m:int(merged[m+"_pm_e"].notna().sum()) for m in ("ours","xiao","nsc","r")},
              strict_three_catalogue=int(merged[["ours_pm_e","xiao_pm_e","nsc_pm_e"]].notna().all(axis=1).sum()),
              private_pair_sha256=digest(final),closure_sha256=digest(closure),input_signature=digest(proof),
              selection="Frozen DR16Q/DESI union; blind and physical-r to QSO 0.5 arcsec, published positional lookup 1 arcsec, "
                        "0.3 arcsec competitor margin and one-to-one claims. Published NSC/Xiao states evaluated after source closure."))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for n in ("root","spectra","xiao","output"):p.add_argument(n,type=Path)
    p.add_argument("--workers",type=int,default=4);a=p.parse_args()
    if not os.getenv("SLURM_JOB_ID") or not 1<=a.workers<=int(os.environ["SLURM_CPUS_PER_TASK"]):p.error("Use assigned Slurm CPUs")
    run(a.root,a.spectra,a.xiao,a.output,a.workers)
