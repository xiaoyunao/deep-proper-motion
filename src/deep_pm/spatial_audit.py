"""Recheck historical source selection and zero points without changing production."""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
import os
from pathlib import Path
from time import monotonic

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from . import access
from .__main__ import tomllib
from .astrometry import project
from .astrometry_audit import calibrate_errors, quality_flags
from .dark_precision import bin_name
from .epoch_statistics import observation_groups
from .io import save_json
from .nsc_comparison import configuration, comparison, qso_matches
from .pilot_production import make_tracks, quality_detections, source_funnel
from .pilot_validation import track_audit
from .precision_v03 import calibration, calibration_groups, fit_with_calibration, paired_change
from .provenance import run_stage
from .regional_cohort import load_output
from .regional_production import prediction
from .regional_validation import audit_qso
from .validation import gaia_labels, groups_by_label, ratio


def diagnostic_summary(values, seed, limit=10):
    n=len(values)
    if n<limit:
        return {"n": n}
    a=np.asarray(values,dtype=float)
    med=np.median(a,axis=0)
    random=np.random.default_rng(seed)
    draws=random.integers(0,n,(400,n))
    ci=np.quantile(np.median(a[draws],axis=1),[.025,.975],axis=0).T
    return dict(n=n,median=med,median95=ci,radial_abs68=np.quantile(np.linalg.norm(a,axis=1),.68))


def reference_gradient(refs, xy, corrected, nsc, support, radius_mas, seed):
    """Fit a centre-referenced linear SDSS residual using training galaxies."""
    coords={}
    for is_holdout, table in refs.loc[support[refs.index]].groupby("is_holdout"):
        positions, gaps=[],[]
        for _,rows in table.groupby("reference",sort=False):
            i=rows.index.to_numpy()
            sd=i[~nsc[i]]
            ns=i[nsc[i]]
            if len(sd) and len(ns):
                positions.append(np.median(xy[ns],axis=0)/radius_mas)
                gaps.append(np.median(corrected[sd],axis=0)-np.median(corrected[ns],axis=0))
        coords[is_holdout]=np.asarray(positions).reshape(-1,2),np.asarray(gaps).reshape(-1,2)
    train,target=coords.get(False,(np.empty((0,2)), np.empty((0,2))))
    test,baseline=coords.get(True,(np.empty((0,2)), np.empty((0,2))))
    if len(train)<60 or len(test)<30 or np.linalg.matrix_rank(train)<2:
        return None,dict(status="insufficient_references",training=len(train),withheld=len(test))
    weight=np.ones(len(train))
    for _ in range(5):
        root=np.sqrt(weight)[:,None]
        slope=np.linalg.lstsq(train*root,target*root,rcond=None)[0]
        radial=np.linalg.norm(target-train@slope,axis=1)
        scale=max(1.,np.median(radial))
        weight=np.minimum(1,2*scale/np.maximum(radial,1e-8))
    updated=baseline-test@slope
    return slope, dict(status="tested",training=len(train),withheld=len(test),
                       coefficients_mas_per_core_radius=slope,
                       held_galaxies=dict(before=diagnostic_summary(baseline,seed),
                                         gradient=diagnostic_summary(updated,seed),
                                         paired=paired_change(baseline,updated,seed)))


def measures_for_truth(truth, xy, xy_nsc_tree, nsc, fwhm, magnitude, morphology):
    result={}
    for label,ix in truth.items():
        i=ix[nsc[ix]]
        if not len(i):
            continue
        p=np.median(xy[i],axis=0)
        # The nearest anchor is the source itself for well-supported matches.
        proximity=xy_nsc_tree.query(p,k=2)[0][1]
        finite=fwhm[i][np.isfinite(fwhm[i])]
        result[label]=(float(np.median(magnitude[i])),float(np.median(morphology[i])),
                       float(np.median(finite)) if len(finite) else None,float(proximity))
    return result


def selection_trials(truth, labels, metadata, products, tracks, nsc, ref, settings):
    prod=settings["production"]
    summary={}
    for name,chosen in products.items():
        matched,global_stats=track_audit(tracks,chosen,truth,labels,nsc,prod["validation_min_recovery"])
        strata=defaultdict(list)
        for label,(r,shape,fwhm,near) in metadata.items():
            if label not in truth:
                continue
            mag=bin_name(r,settings["reference_r_edges"])
            admissible=label in matched
            pm=chosen[matched[label]][1]-ref[label] if admissible else None
            row=(admissible,pm)
            strata[mag].append(row)
            strata[mag+"__"+("seeing_unknown" if fwhm is None else "seeing_le_1p35" if fwhm<=1.35 else "seeing_gt_1p35")].append(row)
            strata[mag+"__"+("neighbor_le_2p5arcsec" if near<=2500 else "neighbor_wider")].append(row)
        bins={}
        for tag,rows in strata.items():
            accepted=[pm for flag,pm in rows if flag]
            bins[tag]=dict(retained=ratio(len(accepted),len(rows)),
                           precision=diagnostic_summary(accepted,settings["seed"],settings["spatial"]["min_publish_slice"]))
        summary[name]=dict(association=global_stats,by_condition=bins)
    return summary


def east_holdout(work, spec, data, xy, t, sigma, keep, nsc, keys, support, refs, offsets, config, settings, custom):
    """Measure held-year geometry separately from a nearby catalogue detection."""
    from matplotlib.path import Path as Polygon

    center=spec["ra"],spec["dec"]
    cache=access.Cache(work/"footprint_cache")
    radius=spec["radius"]+.25
    names=[f"vertex_{axis}{i}" for axis in ("ra","dec") for i in range(1,5)]
    where=access.q3c(*center,radius)+" AND filter='r'"
    chips=access.checked_spatial(cache,"nsc","exposure,"+",".join(names)+",instrument,airmass",
                                  "nsc_dr2.chip",where,center[1]-radius-1e-8,center[1]+radius+1e-8)
    shapes=defaultdict(list)
    corners=chips[names].to_numpy().reshape(-1,2,4)
    ids=np.flatnonzero(np.isfinite(corners).all(axis=(1,2)))
    points=project(corners[ids,0,:].ravel(),corners[ids,1,:].ravel(),center).reshape(-1,4,2)
    for i,polygon in zip(ids,points):
        shapes["NSC:"+str(chips.exposure.iloc[i])].append(Polygon(polygon))
    cam=chips.drop_duplicates("exposure").set_index("exposure").instrument
    instrument=data.exposure_id.str[4:].map(cam).fillna("unknown").str.strip().str.lower()
    # Use the original verified v0.7 exposure metadata when available. The
    # chip query above does not include non-r-only exposure aliases.
    file=work/"exposures.csv"
    if file.is_file():
        meta=pd.read_csv(file).set_index("exposure").instrument
        instrument=data.exposure_id.str[4:].map(meta).fillna("unknown").str.strip().str.lower()
    token=instrument.str.extract(r"^(c4d|k4m|ksb|decam|mosaic3|90prime)$",expand=False).fillna("unknown")
    features=np.where(nsc,"NSC_"+token+"_"+data.band,"SDSS_r")
    groups,_=observation_groups(data)
    raw=data[["sigma_east_mas","sigma_north_mas"]].to_numpy()
    correction=np.array([offsets[k][0] if k in offsets else np.full(2,np.nan) for k in keys])
    w1,_,families=calibrate_errors(data,xy,raw,xy-correction,refs,features,groups,support,custom)
    qualified={k for k,v in families.items() if k.startswith("NSC_") and all(v["calibrated_axes"])}
    expanded=keep | (nsc & data.band.isin(["g","i","z"]).to_numpy()
                     & np.isin(features,list(qualified)) & data.magnitude.notna().to_numpy()
                     & data.magnitude_error.between(0,2.5/(np.log(10)*settings["production"]["min_snr"]),inclusive="right").to_numpy()
                     & quality_flags(data))
    return prediction(work,center,data,xy,t,sigma,w1,keys,nsc,keep,expanded,offsets,
                      settings,config,spec,chip_shapes=shapes)


def run(work, name, spec, config, settings, custom):
    started=monotonic()
    center=spec["ra"],spec["dec"]
    data,xy,t,sigma,keep=quality_detections(work,center,config,settings["production"])
    tracks=[np.asarray(row,dtype=int) for row in json.loads((work/"private_tracks.json").read_text())["tracks"]]
    qso=pd.read_csv(work/"spectroscopic_qso.csv")
    offsets,cal,refs=calibration(work,center,config,project(qso.ra,qso.dec,center),return_references=True,
                                  arrays=(data,xy,t,sigma))
    cal.pop("sdss_run_camcol",None)
    nsc=data.survey.eq("NSC").to_numpy()
    key=calibration_groups(data)
    support=np.isin(key,list(offsets))
    print(f"{name}: {len(offsets)} M3 groups",flush=True)
    initial=make_tracks(xy,data.exposure_id.to_numpy(),nsc&keep,settings["production"]["match_radius_mas"])
    tree=cKDTree(np.array([np.median(xy[ix],axis=0) for ix in initial]).reshape(-1,2))
    variants={}
    cache={}
    trace={}
    for cut in (.6,.8):
        settings_cut=dict(settings,production=dict(settings["production"],min_anchor_morphology=cut))
        anchors={}
        actual=source_funnel(data,xy,t,sigma,keep,tracks,offsets,config,settings_cut,spec["radius"],
                             trace=trace if cut==.8 else None,initial=initial,anchor_info=anchors,model_cache=cache)[1]
        variants[f"{cut:g}"]=actual
        if cut==.6:
            variants.update({f"{v:g}":{j:row for j,row in actual.items() if anchors[j][0]>=v}
                             for v in (.7,.9)})
    detailed=[]
    audit,_=audit_qso(work,center,data,xy,t,nsc,keep,keep,support,{"W0":(tracks,variants["0.8"],trace)},
                      qso,dict(settings,bootstrap_draws=500),details=detailed)
    qso=qso.loc[qso.in_core].reset_index(drop=True)
    positions=project(qso.ra,qso.dec,center)
    qlabels,_=qso_matches(data,xy,keep,positions)
    qtruth={label:index for label,_,index,*_ in detailed}
    fwhm=data.fwhm.to_numpy()
    mag=data.magnitude.to_numpy()
    morph=data.morphology.to_numpy()
    qmeta=measures_for_truth(qtruth,xy,tree,nsc,fwhm,mag,morph)
    qmeta={label:(r,*qmeta[label][1:]) for label,r,*_ in detailed}
    qresult=selection_trials(qtruth,qlabels,qmeta,variants,tracks,nsc,np.zeros((len(qso),2)),settings)
    idx=np.flatnonzero(keep)
    lab,gaia=gaia_labels(data.iloc[idx].reset_index(drop=True),xy[idx],t[idx],sigma[idx],
                        pd.read_csv(work/"validation/gaia.csv"),center,dict(config,gaia_max_pm_error=2.))
    glabels=np.full(len(data),-1)
    glabels[idx]=lab
    gtruth=groups_by_label(glabels,data,t,gaia,center,spec["radius"],config)
    gmeta=measures_for_truth(gtruth,xy,tree,nsc,fwhm,mag,morph)
    motion=gaia[["pm_tangent_x","pm_tangent_y"]].to_numpy()
    gresult=selection_trials(gtruth,glabels,gmeta,variants,tracks,nsc,motion,settings)
    print(f"{name}: {len(qtruth)} strict QSOs, {len(gtruth)} labelled stars",flush=True)
    diagnoses=defaultdict(list)
    private=[]
    when=data.observation_time.to_numpy()
    exposure=data.exposure_id.to_numpy()
    corrected=xy-np.array([offsets[k][0] if k in offsets else np.full(2,np.nan) for k in key])
    radius=np.tan(np.deg2rad(spec["radius"]))*180/np.pi*3_600_000
    priority=name in {"north_c","east_d","west_a","crowd_a"}
    slope,galaxy=reference_gradient(refs,xy,corrected,nsc,support,radius,settings["seed"]) if priority else (None,{"status":"not_in_frozen_gradient_test"})
    contrast=defaultdict(list)
    inst={}
    if (work/"exposures.csv").is_file():
        meta=pd.read_csv(work/"exposures.csv").drop_duplicates("exposure")
        inst={"NSC:"+str(row.exposure):str(row.instrument).strip() for row in meta.itertuples()}
    color_rows={}
    for band,band_label in (("g","g"),("i","i")):
        selection=nsc & data.band.eq(band).to_numpy() & data.magnitude_error.between(0,.22,inclusive="right").to_numpy()
        _, nearby=qso_matches(data,xy,selection,positions)
        color_rows[band_label]={k:np.median(mag[v]) for k,v in nearby.items()}
    for label,r,_,slot,reason,est in detailed:
        if slot is None:
            continue
        _,mu,err,_,use,_=variants["0.8"][slot]
        early=use[~nsc[use]]
        late=use[nsc[use]]
        model=cache[slot][1]
        dt=t[use,None]-np.average(np.repeat(t[use,None],2,axis=1),axis=0,weights=model.weight[:,None]/sigma[use]**2)
        amp=np.linalg.norm(model.weight[:,None]*dt/sigma[use]**2,axis=1)
        eidx=np.flatnonzero(~nsc[use])
        dominant=key[use[eidx[np.argmax(amp[eidx])]]]
        # Only aggregate identifiers for instrument and hashed scan/camera.
        camera=sha256(str(dominant).encode()).hexdigest()[:7]
        instr=Counter(inst.get(e,"unknown") for e in exposure[late]).most_common(1)[0][0]
        color=color_rows["g"].get(label,np.nan)-color_rows["i"].get(label,np.nan)
        magbin=bin_name(r,settings["reference_r_edges"])
        cbin="missing" if not np.isfinite(color) else "blue" if color<.5 else "red" if color>=1.2 else "middle"
        age="short" if np.ptp(t[late])<2 else "long"
        era="early" if np.median(when[late])<57753 else "late"
        for k in ("magnitude__"+magbin,"color__"+cbin,"instrument__"+instr,"scan_camera__"+camera,
                  "nsc_span__"+age,"nsc_time__"+era,"matched_condition__"+"__".join((camera,instr,age,era,magbin))):
            diagnoses[k].append(mu)
        private.append((r, camera, instr, age, era, cbin, mu))
        if slope is not None:
            trial=xy[use].copy()
            trial[~nsc[use]]-=trial[~nsc[use]]/radius@slope
            corrected_model,new_error=fit_with_calibration(trial,t[use],sigma[use],config["reference_epoch"],
                                                        [offsets[k] for k in key[use]])
            row=dict(W0=(mu,err),gradient=(corrected_model.coeff[1],new_error))
            contrast["all"].append(row)
            contrast[magbin].append(row)
    if contrast:
        opts=dict(settings,bootstrap_draws=600)
        galaxy["independent_qso"]=dict((k,comparison(v,opts,(("gradient","W0"),))) for k,v in contrast.items())
    held=east_holdout(work,spec,data,xy,t,sigma,keep,nsc,key,support,refs,offsets,config,settings,custom) if name=="east_c" else {}
    save_json(work/"private_diagnostics.json",private)
    result=dict(qso=audit, morphology=dict(qso=qresult,gaia=gresult,
                 catalogue_sizes={k:len(v) for k,v in variants.items()}),
                zero={k:diagnostic_summary(v,settings["seed"],settings["spatial"]["min_publish_slice"])
                      for k,v in diagnoses.items()}, local_gradient=galaxy, holdout_geometry=held,
                m3=cal,resources={"wall_seconds":monotonic()-started})
    save_json(work/"spatial_audit_summary.json",result)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("work",type=Path)
    p.add_argument("region")
    p.add_argument("prior",type=Path)
    p.add_argument("--config",type=Path,default=Path("configs/spatial-v08.toml"))
    args=p.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        p.error("Use a Slurm compute node.")
    config,settings=configuration(args.config)
    custom=tomllib.loads(args.config.with_name("astrometry-v06.toml").read_text())
    old=args.config.with_name("regional-v07.toml")
    spec=configuration(old)[1]["fields"][args.region]
    load_output(args.prior,"regional-production-v07","regional_summary.json")
    work=args.work
    work.mkdir(parents=True,exist_ok=True)
    names=("detections.csv","external_labels.csv","official_nsc.csv","spectroscopic_qso.csv",
           "private_tracks.json","validation/gaia.csv","exposures.csv","run_manifest.json")
    links={name:args.prior/name for name in names}
    links["prior_manifest.json"]=links.pop("run_manifest.json")
    links["prior_config.toml"]=old
    links["m3_model.toml"]=args.config.with_name(settings["frozen_model"])
    for name,path in links.items():
        out=work/name
        out.parent.mkdir(parents=True,exist_ok=True)
        target=path.resolve(strict=True)
        if not out.exists():
            out.symlink_to(target)
        if out.resolve()!=target:
            p.error("Use a separate receipt directory when upstream inputs change.")
    with run_stage(work,args.config,args.region,"spatial-audit-v08",inputs=list(links),
                   outputs=["spatial_audit_summary.json","private_diagnostics.json"]) as go:
        if go:
            run(work,args.region,spec,config,settings,custom)


if __name__=="__main__":
    main()
