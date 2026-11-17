"""Compare frozen M3 and detector-scale offsets on the same real detections."""
import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
from time import monotonic

import healpy as hp
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .__main__ import tomllib
from .astrometry import project, unproject
from .dark_precision import query_unique
from .exposure_catalogues import inside
from .exposure_model import hierarchy, observation_keys
from .io import save_json
from .nsc_comparison import comparison, qso_matches
from .paired_control import catalogue_motion, common_catalogue_ids
from .pilot_production import make_tracks, quality_detections, source_funnel, valid_epochs
from .pilot_validation import track_audit
from .precision_v03 import calibration, fit_with_calibration
from .provenance import run_stage
from .regional_validation import official_table, tag_magnitude, valid_published
from .validation import gaia_labels, groups_by_label


def matched_legacy(table, points, center):
    data=table.rename(columns=str.lower)
    idx=np.flatnonzero(np.isfinite(data[["ra","dec"]]).all(axis=1))
    if not len(idx):
        return {},{}
    found,neighbors=query_unique(cKDTree(project(data.ra.iloc[idx],data.dec.iloc[idx],center)),
                                  points,1000,300)
    result={}
    categories=Counter()
    for k in np.flatnonzero(found):
        row=data.iloc[idx[neighbors[k]]]
        label=str(row.pm_source).strip()
        categories[label]+=1
        error=row[["pmra_err","pmdec_err"]].to_numpy(dtype=float)
        output=catalogue_motion(row,center)
        if label=="DESI" and np.isfinite(output).all() and np.isfinite(error).all() and (error>0).all():
            result[k]=(output,error)
    return result,categories


def held_galaxies(refs,xy,nsc,keys,offsets,pixels,fine,settings):
    a,b=keys
    c0,c1=offsets
    use=refs.is_holdout & np.isin(a[refs.index],list(c0)) & np.isin(b[refs.index],list(c1))
    rows=refs.loc[use]
    points=[]
    boundary=defaultdict(list)
    for _,row in rows.groupby("reference",sort=False):
        i=row.index.to_numpy()
        early,late=i[~nsc[i]],i[nsc[i]]
        if not len(early) or not len(late):
            continue
        x0=xy[i]-np.stack([c0[a[j]][0] for j in i])
        x1=xy[i]-np.stack([c1[b[j]][0] for j in i])
        values=np.array([np.median(x[~nsc[i]],axis=0)-np.median(x[nsc[i]],axis=0) for x in (x0,x1)])
        points.append(dict(C0=(values[0],np.ones(2)),C1=(values[1],np.ones(2))))
        # A matched galaxy seen by different detectors of the same parent probes an actual edge.
        for group in set(a[i]):
            q=np.flatnonzero(a[i]==group)
            if len(set(fine[i[q]]))>1:
                boundary["physical_detector"].append([
                    np.max(np.linalg.norm(x[q]-np.median(x[q],axis=0),axis=1)) for x in (x0,x1)])
        if len(set(pixels[i]))>1:
            boundary["cross_pixel_position"].append(np.linalg.norm(values,axis=1))
    return dict(comparison=comparison(points,dict(settings,tail_masyr=150),(("C1","C0"),)),
                boundary={k:dict(n=len(v),p68_mas=np.quantile(v,.68,axis=0)) for k,v in boundary.items()})


def contrast_samples(truth, labels, data, xy, t, sigma, keep, nsc, settings, config,
                     keys, offsets, points, magnitudes, origin, ref, ref_error,
                     release, legacy, tracks, products):
    truth={k:i for k,i in truth.items() if nsc[i].any() and not nsc[i].all()}
    linked,status=common_catalogue_ids(truth,labels,data,release[0])
    published=release[1]
    ceilings={k:track_audit(tracks,values,truth,labels,nsc,
                   settings["production"]["validation_min_recovery"])[0] for k,values in products.items()}
    support=np.isin(keys[0],list(offsets[0]))&np.isin(keys[1],list(offsets[1]))
    records=defaultdict(lambda:defaultdict(list))
    totals=Counter()
    for k,detections in truth.items():
        if k>=len(origin) or not np.isfinite(magnitudes[k]):
            continue
        indices=detections[support[detections]&keep[detections]]
        if not valid_epochs(indices,nsc,t,settings["production"]):
            continue
        base={}
        for name,keyset,solution in zip(("C0","C1"),keys,offsets):
            fit,err=fit_with_calibration(xy[indices],t[indices],sigma[indices],config["reference_epoch"],
                                        [solution[keyset[i]] for i in indices])
            base[name]=(fit.coeff[1],err)
        if linked.get(k) in published.index:
            values=valid_published(published.loc[linked[k]],points)
            if values is not None:
                base["NSC"]=values
        if k in legacy:
            base["Xiao"]=legacy[k]
        blind={name:products[name][found[k]][1:3] for name,found in ceilings.items() if k in found}
        if all(name in blind for name in ("C0","C1")):
            blind.update({name:x for name,x in base.items() if name in ("Xiao","NSC")})
        else:
            blind={}
        totals["strict"]+=1
        totals["strict_all_catalogues"]+=len(base)==4
        totals["blind_both_models"]+=bool(blind)
        totals["blind_all_catalogues"]+=len(blind)==4
        for state,values in (("strict",base),("blind",blind)):
            if not values:
                continue
            corrected={name:(pm-ref[k],np.hypot(error,ref_error[k])) for name,(pm,error) in values.items()}
            for mag in tag_magnitude(magnitudes[k],settings):
                for source in {"all",origin[k]}:
                    prefix=f"{state}__{source}__{mag}"
                    records["c0_c1"][prefix].append({name:corrected[name] for name in ("C0","C1")})
                    if len(values)==4:
                        records["all_four"][prefix].append(corrected)
    comparisons=(("C1","C0"),("C1","NSC"),("C1","Xiao"),("C0","Xiao"),("C0","NSC"))
    return dict(totals=totals,archive_match_status=status,by_cohort={kind:{
        label:comparison(r,settings,comparisons) for label,r in bins.items()} for kind,bins in records.items()})


def run(work,inputs,legacy_path,spec,calibration_config):
    start=monotonic()
    settings=tomllib.loads(Path("configs/spatial-v08.toml").read_text())
    settings["bootstrap_draws"]=400
    config=tomllib.loads(Path("configs/precision-v03.toml").read_text())
    center=tuple(np.mean(spec[k]) for k in ("ra","dec"))
    radius=1.5
    data,xy,t,sigma,keep=quality_detections(inputs,center,config,settings["production"])
    nsc=data.survey.eq("NSC").to_numpy()
    fine=observation_keys(data)[1]
    pixels=hp.ang2pix(128,data.ra_deg.to_numpy(),data.dec_deg.to_numpy(),lonlat=True)
    qso=pd.read_csv(inputs/"spectra_union.csv").query("in_core").reset_index(drop=True)
    veto=pd.read_csv(inputs/"spectra_veto.csv")
    c0,c0_stats,refs=calibration(inputs,center,config,project(veto.ra,veto.dec,center),
                                return_references=True,arrays=(data,xy,t,sigma))
    c0_stats.pop("sdss_run_camcol",None)
    print(f"C0: {len(c0)} calibrated groups",flush=True)
    keys,c1,c1_stats=hierarchy(data,refs,xy,sigma,config,calibration_config,c0)
    pairs=(observation_keys(data)[0],keys)
    models=(c0,c1)
    galaxy=held_galaxies(refs,xy,nsc,pairs,models,pixels,fine,settings)
    print(f"C1: {c1_stats}; held galaxies {galaxy['comparison']['n']}",flush=True)
    visits=data.exposure_id.to_numpy()
    tracks=make_tracks(xy,visits,keep,1000.)
    seeds=make_tracks(xy,visits,keep&nsc,1000.)
    products={}
    yields={}
    for name,obs,solution in zip(("C0","C1"),pairs,models):
        def core(position):
            ra,dec=unproject(position,center)[0]
            return inside(ra,dec,spec)
        yields[name],products[name]=source_funnel(data,xy,t,sigma,keep,tracks,solution,config,settings,radius,
            initial=seeds,observation_groups=obs,core_members=core)
    original=pd.read_csv(inputs/"external_labels.csv",dtype=str).set_index("detection_id").original_object_id
    release=(original,official_table(inputs,{}))
    legacy=pd.read_csv(legacy_path)
    legacy=legacy.loc[inside(legacy.RA,legacy.DEC,spec,.06)]
    projected=project(qso.ra,qso.dec,center)
    labels,matches=qso_matches(data,xy,keep,projected)
    old,old_count=matched_legacy(legacy,projected,center)
    qso_stats=contrast_samples(matches,labels,data,xy,t,sigma,keep,nsc,settings,config,
        pairs,models,center,qso.reference_r.to_numpy(),qso.origin.to_numpy(),
        np.zeros((len(qso),2)),np.zeros((len(qso),2)),release,old,tracks,products)
    print(f"QSO controls: {qso_stats['totals']}",flush=True)
    g=pd.read_csv(inputs/"validation/gaia.csv")
    labels0,cat=gaia_labels(data.loc[keep].reset_index(drop=True),xy[keep],t[keep],sigma[keep],
                           g,center,dict(config,gaia_max_pm_error=2.))
    labels=np.full(len(data),-1)
    labels[keep]=labels0
    truth=groups_by_label(labels,data,t,cat,center,radius,config)
    truth={i:idx for i,idx in truth.items() if inside(cat.ra.iloc[i],cat.dec.iloc[i],spec)}
    mag=np.full(len(cat),np.nan)
    for i,idx in truth.items():
        mag[i]=np.median(data.magnitude.iloc[idx[nsc[idx]]])
    old_gaia,old_gaia_counts=matched_legacy(legacy,project(cat.ra,cat.dec,center),center)
    gaia_stats=contrast_samples(truth,labels,data,xy,t,sigma,keep,nsc,settings,config,
        pairs,models,center,mag,np.repeat("Gaia",len(cat)),cat[["pm_tangent_x","pm_tangent_y"]].to_numpy(),
        cat[["pmra_error","pmdec_error"]].to_numpy(),release,old_gaia,tracks,products)
    save_json(work/"pilot_summary.json",dict(c0=c0_stats,c1=c1_stats,galaxies=galaxy,qso=qso_stats,
        gaia=gaia_stats,legacy_origin_counts=dict(qso=old_count,gaia=old_gaia_counts),source_yield=yields,
        qso_origins=qso.origin.value_counts(),elapsed_seconds=monotonic()-start,
        acquisition=json.loads((inputs/"acquisition_summary.json").read_text())))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("root",type=Path)
    p.add_argument("region")
    p.add_argument("inputs",type=Path)
    p.add_argument("legacy",type=Path)
    p.add_argument("config",type=Path)
    a=p.parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        p.error("Run on a Slurm compute node")
    settings=tomllib.loads(a.config.read_text())
    work=a.root/a.region
    work.mkdir(parents=True,exist_ok=True)
    with run_stage(work,a.config,a.region,"exposure-pilot-v09",outputs=["pilot_summary.json"]) as go:
        if go:
            run(work,a.inputs/a.region,a.legacy,settings["fields"][a.region],settings["calibration"])


if __name__=="__main__":
    main()
