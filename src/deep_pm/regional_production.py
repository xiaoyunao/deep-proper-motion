"""Regional W0 production and matched W1/multiband validation."""
import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
from time import monotonic, process_time

from astropy.time import Time
from matplotlib.path import Path as Polygon
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .__main__ import tomllib
from . import access
from .astrometry import fit_motion, project, unproject, utc_to_tt, years
from .astrometry_audit import augment_tracks, calibrate_errors, exposure_metadata, quality_flags, qtile
from .dark_precision import query_unique
from .epoch_statistics import observation_groups, source_coverage
from .io import digest, save_json
from .nsc_comparison import configuration
from .pilot_production import make_tracks, quality_detections, source_funnel, timing_resources, valid_epochs
from .precision_v03 import calibration, calibration_groups
from .provenance import run_stage
from .regional_preflight import CHIP_VERTICES, r_chips
from .regional_validation import audit_gaia, audit_qso
from .validation import ratio


def blind_products(data,xy,t,sigma,w1,nsc,keep,expanded,tracks,offsets,config,settings,spec,fits=None):
    samples={}
    outputs={}
    expanded_tracks=augment_tracks(tracks,data,xy,expanded,keep,settings["production"])
    initial=make_tracks(xy,data.exposure_id.to_numpy(),keep & nsc,settings["production"]["match_radius_mas"])
    for name,error,flags,partition in (("W0",sigma,keep,tracks),("W1",w1,keep,tracks),
                                       ("multi",w1,expanded,expanded_tracks)):
        trace={}
        funnel,final=source_funnel(data,xy,t,error,flags,partition,offsets,config,settings,spec["radius"],
                                  reference_keep=keep,trace=trace,fits=fits if name=="W0" else None,initial=initial)
        samples[name]=partition,final,trace
        outputs[name]=funnel
    return samples,outputs


def record_catalogue(work,center,data,xy,t,sigma,keys,groups,night,nsc,offsets,products,fits,config,settings):
    flags=quality_flags(data)
    visits=data.exposure_id.to_numpy()
    correction=np.array([offsets[k][0] if k in offsets else np.zeros(2) for k in keys])
    coord=xy-correction
    records=[]
    leverage=[]
    epoch=config["reference_epoch"]
    for name,(_,solutions,_) in products.items():
        for slot,(pos,pm,err,tag,used,n) in solutions.items():
            early=used[~nsc[used]]
            late=used[nsc[used]]
            radec=unproject(pos,center)[0]
            n_run=len(set(visits[early]))
            n_groups=len(set(groups[used]))
            n_flags=int((~flags[used]).sum())
            influence=shift=None
            downweighted=None
            if name=="W0":
                model=fits[slot]
                w=model.weight[:,None]/sigma[used]**2
                dt=t[used,None]-np.average(np.repeat(t[used,None],2,axis=1),axis=0,weights=w)
                weight=np.linalg.norm(w*dt/np.sum(w*dt**2,axis=0),axis=1)
                by_group={key:float(weight[groups[used]==key].sum()) for key in set(groups[used])}
                influence=max(by_group.values())/weight.sum()
                for group in sorted(by_group,key=lambda k:(-by_group[k],k)):
                    trimmed=used[groups[used]!=group]
                    if valid_epochs(trimmed,nsc,t,settings["production"]):
                        fit=fit_motion(coord[trimmed],t[trimmed],sigma[trimmed],epoch)
                        shift=float(np.linalg.norm(fit.coeff[1]-pm))
                        break
                downweighted=int((model.weight<.5).sum())
                leverage.append((n_run,n_groups,n_flags,downweighted,influence,shift))
            records.append((name,slot,*radec,*pm,*err,tag,len(used),n,n_run,len(set(night[early])),
                            np.ptp(t[early]),len(set(night[late])),np.ptp(t[late]),np.ptp(t[used]),
                            n_groups,n_flags,downweighted,influence,shift))
    columns=("method","track_id","ra","dec","pm_east","pm_north","pm_error_east","pm_error_north","r_bin",
             "fit_detections","associated_detections","sdss_runs","sdss_nights","sdss_span_years",
             "nsc_nights","nsc_span_years","joint_span_years","independent_groups",
             "detector_flags","downweighted","max_group_slope_fraction","delete_group_pm_shift")
    pd.DataFrame(records,columns=columns).to_csv(work/"private_catalogue.csv.gz",index=False)
    result={}
    for key,rows in (("all",leverage),("one_sdss",[v for v in leverage if v[0]==1]),
                     ("repeat_sdss",[v for v in leverage if v[0]>=2])):
        shifts=[v[-1] for v in rows if v[-1] is not None]
        result[key]=dict(n=len(rows),shift_supported=ratio(len(shifts),len(rows)),
                         independent_groups_flags_downweight_leverage_p16_50_84=qtile([v[1:5] for v in rows]),
                         pm_shift_p16_50_84=qtile(shifts))
    return result


def r_chip_shapes(chips, center):
    shapes=defaultdict(list)
    vertices=chips[CHIP_VERTICES].to_numpy().reshape(-1,2,4)
    valid=np.isfinite(vertices).all(axis=(1,2)) & chips.mjd.between(50000,62000)
    ids=np.flatnonzero(valid)
    xy=project(vertices[ids,0,:].ravel(),vertices[ids,1,:].ravel(),center).reshape(-1,4,2)
    for row,corners in zip(ids,xy):
        shapes[chips.exposure.iloc[row],chips.mjd.iloc[row]].append(Polygon(corners))
    epochs=years(utc_to_tt(np.array([mjd for _,mjd in shapes])))
    return list(zip(epochs,shapes.values()))


def prediction(work,center,data,xy,t,sigma,w1,keys,nsc,keep,expanded,offsets,settings,config,spec,chip_shapes=None):
    record_extra=spec.get('record_extra_sources',False)
    if record_extra:save_json(work/'private_extra_training_sources.json',[])
    # Choose the most observed interior calendar year by survey metadata alone.
    time_edges=Time([f"{y}-01-01" for y in range(2010,2021)],scale="utc").tt.jyear
    survey_year=np.searchsorted(time_edges,t,side="right")-1
    calibrated=np.isin(keys,list(offsets))
    usable=nsc & keep & calibrated
    years=np.unique(survey_year[usable])
    if len(years)<3:
        return dict(status="insufficient_calibrated_nsc_years",n_years=len(years))
    exp=data.exposure_id.to_numpy()
    year=max(years[1:-1],key=lambda y:len(set(exp[usable & (survey_year==y)])))
    hold=nsc & (survey_year==year)
    prod=settings["production"]
    fresh=make_tracks(xy,exp,keep & ~hold,prod["match_radius_mas"])
    initial=make_tracks(xy,exp,keep & ~hold & nsc,prod["match_radius_mas"])
    fresh_multi=augment_tracks(fresh,data,xy,expanded & ~hold,keep & ~hold,prod)
    core=spec.get("core_members")
    baseline=source_funnel(data,xy,t,w1,keep & ~hold,fresh,offsets,config,settings,spec["radius"],initial=initial,
                           observation_groups=keys,core_members=core)[1]
    extra_models={} if record_extra else None
    extra=source_funnel(data,xy,t,w1,expanded & ~hold,fresh_multi,offsets,config,settings,spec["radius"],
                        reference_keep=keep & ~hold,initial=initial,observation_groups=keys,core_members=core,fits=extra_models)[1]
    if record_extra:
        from .standard_motion import calibrated_covariance,icrs_motion
        records=[]
        for slot,value in extra.items():
            if slot in baseline:continue
            use=value[4]
            covariance=calibrated_covariance(extra_models[slot],t[use],w1[use],[offsets[keys[j]] for j in use])[0]
            records.append(dict(r_bin=value[3],**icrs_motion(value[0],value[1],center,covariance),
                detection_ids=data.detection_id.iloc[use].tolist(),exposures=sorted(set(exp[use])),
                nsc_epoch_days=sorted(set(np.floor((t[use[nsc[use]]]-2000)*365.25).astype(int)))))
        save_json(work/'private_extra_training_sources.json',records)
    # Use W1 as the common anchor. A single observed position is compared to
    # both predictions. Newly supported sources are a distinct denominator.
    pool=[j for j,v in extra.items() if v[3] in ("20_to_21","21_to_22","22_to_23")]
    if not pool:
        return dict(status="no_eligible_held_sources",year=int(year+2010))
    route=[extra[j] for j in pool]
    key_pos=np.array([baseline.get(j,extra[j])[0] for j in pool])
    key_pm=np.array([baseline.get(j,extra[j])[1] for j in pool])
    other_pos=np.array([v[0] for v in route])
    other_pm=np.array([v[1] for v in route])
    compare=[("common" if j in baseline else "new",extra[j][3]) for j in pool]
    counts=Counter(compare)
    pairs=defaultdict(list)
    opportunities=Counter()
    outside=Counter()
    shifted=defaultdict(set)
    null_shifts=[np.array(v) for v in ((30000.,10000.),(-30000.,10000.),(10000.,-30000.),(-10000.,30000.))] if spec.get('position_null_control',False) else []
    epoch=config["reference_epoch"]
    near=prod["competitor_margin_mas"]
    footprint_sources=np.zeros(len(pool),dtype=bool)
    opportunity_sources=np.zeros(len(pool),dtype=bool)
    chip_visits=0
    if chip_shapes is not None:
        for midpoint,polygons in chip_shapes:
            if time_edges[year]<=midpoint<time_edges[year+1]:
                chip_visits+=1
                pred=key_pos+key_pm*(midpoint-epoch)
                for polygon in polygons:
                    footprint_sources |= polygon.contains_points(pred)
    for _, frame in data.loc[usable & hold].groupby("exposure_id",sort=False):
        ii=frame.index.to_numpy()
        midpoint=np.median(t[ii])
        # Positions corrected onto the same reference frame using held-field
        # galaxies. No stellar detections in this year enter either association.
        coords=xy[ii]-np.array([offsets[k][0] for k in keys[ii]])
        tree=cKDTree(coords)
        pred=key_pos+key_pm*(midpoint-epoch)
        distance=tree.query(pred)[0]
        opportunity_sources |= distance <= 1000
        accepted,closest=query_unique(tree,pred,500,near)
        for shift_id,shift in enumerate(null_shifts):
            null,_=query_unique(tree,pred+shift,500,near)
            for j in np.flatnonzero(null):shifted[(compare[j],shift_id)].add(int(j))
        for j in np.flatnonzero((distance<=1000)&~accepted):
            outside[compare[j]]+=1
        for j in np.flatnonzero(distance<=1000):
            opportunities[compare[j]]+=1
        for j in np.flatnonzero(accepted):
            i=closest[j]
            delta=t[ii[i]]-epoch
            a=np.linalg.norm(coords[i]-key_pos[j]-key_pm[j]*delta)
            b=np.linalg.norm(coords[i]-other_pos[j]-other_pm[j]*delta)
            pairs[compare[j]].append((j,a,b))
    output={}
    for (kind,tag),n in sorted(counts.items()):
        key=kind,tag
        part=pairs[key]
        nstar=len({v[0] for v in part})
        stats=dict(train_sources=n,matched_sources=ratio(nstar,n),
                   near_r_detections=opportunities[key],not_unique_or_outside_500=outside[key],
                   unique_held_positions=len(part))
        if null_shifts:
            stats['shifted_position_null']=[ratio(len(shifted[(key,j)]),n) for j in range(len(null_shifts))]
        if chip_shapes is not None:
            ii=np.array([j for j,k in enumerate(compare) if k==key],dtype=int)
            matched=np.zeros(len(pool),dtype=bool)
            matched[[v[0] for v in part]]=True
            stats["chip_geometry"]=dict(covered=int(footprint_sources[ii].sum()),
                covered_and_detected_within_1arcsec=int((footprint_sources & opportunity_sources)[ii].sum()),
                inside_without_nearby_detection=int((footprint_sources & ~opportunity_sources)[ii].sum()),
                geometric_or_archive_unknown=int((~footprint_sources)[ii].sum()),
                outside_polygon_but_unique=int((~footprint_sources & matched)[ii].sum()),
                near_but_no_unique_match=int((opportunity_sources & ~matched)[ii].sum()))
        if nstar>=settings["min_publish_count"]:
            a=np.asarray(part)[:,1:]
            stats["multi_p68_mas"]=np.quantile(a[:,1],.68)
            if kind=="common":
                stats["common_w1_p68_mas"]=np.quantile(a[:,0],.68)
        output.setdefault(kind,{})[tag]=stats
    result=dict(status="completed",year=int(year+2010),test_r_visits=len(set(exp[usable&hold])),by_r=output)
    if null_shifts:
        result['null_scope']='Four fixed 31.6 arcsec position shifts at actual held times. Held-position chance coincidence only; not a whole-track contamination estimate or absolute PM truth.'
    if chip_shapes is not None:
        result["visits_with_chip_geometry"]=chip_visits
    return result


def compute(work,region,spec,config,settings,custom,legacy,footprint_cache=None):
    start,cpu=monotonic(),process_time()
    prod=settings["production"]
    center=(spec["ra"],spec["dec"])
    data,xy,t,sigma,keep=quality_detections(work,center,config,prod)
    archive=json.loads((work/"private_tracks.json").read_text())
    tracks=[np.asarray(index,dtype=int) for index in archive["tracks"]]
    qso=pd.read_csv(work/"spectroscopic_qso.csv")
    offsets,calibration_info,refs=calibration(work,center,config,project(qso.ra,qso.dec,center),
                                            return_references=True,arrays=(data,xy,t,sigma))
    calibration_info.pop("sdss_run_camcol",None)
    keys=calibration_groups(data)
    assigned=np.array([offsets[key][0] if key in offsets else np.full(2,np.nan) for key in keys])
    support=np.isfinite(assigned).all(axis=1)
    nsc=data.survey.eq("NSC").to_numpy()
    print(f"{region}: M3 supports {len(offsets)} groups",flush=True)
    settings=dict(settings,bootstrap_draws=min(settings["bootstrap_draws"],1200))
    if legacy:
        trace={}
        funnel,chosen=source_funnel(data,xy,t,sigma,keep,tracks,offsets,config,settings,spec["radius"],trace=trace)
        result,_=audit_qso(work,center,data,xy,t,nsc,keep,keep,support,{"W0":(tracks,chosen,trace)},qso,settings)
        save_json(work/"legacy_exit_summary.json",result)
        return
    meta=pd.read_csv(work/"exposures.csv").set_index("exposure")
    tokens=data.exposure_id.str[4:].map(meta.instrument).fillna("unknown").str.strip().str.lower()
    token=tokens.str.extract(r"^(c4d|k4m|ksb|decam|mosaic3|90prime)$",expand=False).fillna("unknown")
    features=np.where(nsc,"NSC_"+token+"_"+data.band,"SDSS_r")
    groups,night=observation_groups(data)
    raw=data[["sigma_east_mas","sigma_north_mas"]].to_numpy()
    w1,_,errors=calibrate_errors(data,xy,raw,xy-assigned,refs,features,groups,support,custom)
    certified={k for k,v in errors.items() if k.startswith("NSC_") and all(v["calibrated_axes"])}
    expanded=keep | (nsc & data.band.isin(["g","i","z"]).to_numpy()
                     & np.isin(features,list(certified)) & data.magnitude.notna().to_numpy()
                     & data.magnitude_error.between(0,2.5/(np.log(10)*prod["min_snr"]),inclusive="right").to_numpy()
                     & quality_flags(data))
    fits={}
    products,funnel=blind_products(data,xy,t,sigma,w1,nsc,keep,expanded,tracks,offsets,config,settings,spec,fits)
    diagnostics=record_catalogue(work,center,data,xy,t,sigma,keys,groups,night,nsc,offsets,products,fits,config,settings)
    qso_stats,pairs=audit_qso(work,center,data,xy,t,nsc,keep,expanded,support,products,qso,settings)
    save_json(work/"private_qso_pairs.json",pairs)
    print(f"{region}: QSO and private catalogues complete",flush=True)
    gaia_pairs=[] if settings.get("run_series")=="v08" else None
    gaia=audit_gaia(work,center,spec,data,xy,t,sigma,keep,expanded,nsc,products,settings,config,paired_rows=gaia_pairs)
    if gaia_pairs is not None:
        save_json(work/"private_gaia_pairs.json",gaia_pairs)
    chips=None if footprint_cache is None else r_chip_shapes(
        r_chips(access.Cache(footprint_cache),center,spec["radius"]),center)
    holdout=prediction(work,center,data,xy,t,sigma,w1,keys,nsc,keep,expanded,
                       offsets,settings,config,spec,chip_shapes=chips)
    w=products["W1"][1]
    c=products["multi"][1]
    gained, lost=Counter(c[j][3] for j in c.keys()-w.keys()),Counter(w[j][3] for j in w.keys()-c.keys())
    interval=defaultdict(list)
    for j in w.keys()&c.keys():
        tag=w[j][3]
        a=w[j][4]
        b=c[j][4]
        a,b=a[nsc[a]],b[nsc[b]]
        interval[tag].append((len(set(night[b]))-len(set(night[a])),np.ptp(t[b])-np.ptp(t[a])))
    summary=dict(funnel=funnel,qso=qso_stats,gaia=gaia,holdout=holdout,astrometry=diagnostics,
                 calibration=calibration_info,coverage=source_coverage(data,xy,tracks,nsc,keep,night,support,spec,settings),
                 multiband=dict(gained=gained,lost=lost,common_extra_nights_and_span={k:qtile(v) for k,v in interval.items()},
                                fitted_band_groups=sorted(k for k in certified if k.rsplit("_",1)[-1] in ("g","r","i","z"))),
                 resources=timing_resources(work,start,cpu),
                 acquisition=json.loads((work/"acquisition_summary.json").read_text()),
                 external_dark_stars="no_verified_independent_reference_in_current_footprints")
    save_json(work/"regional_summary.json",summary)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("work",type=Path)
    p.add_argument("region")
    p.add_argument("inputs",type=Path)
    p.add_argument("--legacy",action="store_true")
    p.add_argument("--preflight",type=Path)
    p.add_argument("--config",type=Path,default=Path("configs/regional-v07.toml"))
    args=p.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        p.error("Use a Slurm compute node.")
    config,settings=configuration(args.config)
    spec=(configuration(args.config.with_name("production-v05.toml"))[1] if args.legacy else settings)["fields"][args.region]
    custom=tomllib.loads(args.config.with_name("astrometry-v06.toml").read_text())
    work=args.work
    work.mkdir(parents=True,exist_ok=True)
    links={key:args.inputs/key for key in ("detections.csv","external_labels.csv","spectroscopic_qso.csv","validation/gaia.csv")}
    links["m3_config.toml"]=args.config.with_name(settings["frozen_model"])
    links["weight_config.toml"]=args.config.with_name("astrometry-v06.toml")
    if not args.legacy:
        links.update({key:args.inputs/key for key in ("official_nsc.csv","acquisition_summary.json")})
    if args.preflight is not None:
        from .regional_cohort import load_output

        stage="regional-preflight-"+settings["run_series"]
        load_output(args.preflight,stage,"regional_preflight_summary.json")
        links["preflight_manifest.json"]=args.preflight/"run_manifest.json"
        links["regional_preflight_summary.json"]=args.preflight/"regional_preflight_summary.json"
        if settings["run_series"]=="v08":
            matrix=args.config.with_name("spatial-v08-fields.toml")
            locked=tomllib.loads(matrix.read_text())["fields"].get(args.region,{})
            if (locked.get("geometry_sha256")!=digest(links["regional_preflight_summary.json"])
                    or any(spec[k]!=locked.get(k) for k in ("ra","dec","radius"))):
                p.error("Production geometry differs from that frozen before reduction.")
            links["field_matrix.toml"]=matrix
    for name,target in links.items():
        path=work/name
        path.parent.mkdir(parents=True,exist_ok=True)
        target=target.resolve(strict=True)
        if not path.exists():
            path.symlink_to(target)
        if path.resolve()!=target:
            p.error("Use a separate run directory when inputs change.")
    series=settings.get("run_series","v07")
    with run_stage(work,args.config,args.region,"blind-r-"+series,inputs=["detections.csv","m3_config.toml"],
                   outputs=["private_tracks.json"]) as go:
        if go:
            center=spec["ra"],spec["dec"]
            data,xy,_,_,keep=quality_detections(work,center,config,settings["production"])
            tracks=make_tracks(xy,data.exposure_id.to_numpy(),keep,settings["production"]["match_radius_mas"])
            save_json(work/"private_tracks.json",dict(tracks=tracks))
    stage=("legacy-exit-" if args.legacy else "regional-production-")+series
    outputs=["legacy_exit_summary.json"] if args.legacy else [
           "regional_summary.json","private_catalogue.csv.gz","private_qso_pairs.json"]
    if series=="v08" and not args.legacy:
        outputs.append("private_gaia_pairs.json")
    inputs=[*links,"private_tracks.json"]
    if not args.legacy:
        with run_stage(work,args.config,args.region,"exposure-metadata-"+series,inputs=["detections.csv"],
                       outputs=["exposures.csv"]) as go:
            if go:
                frame=pd.read_csv(work/"detections.csv",usecols=["survey","exposure_id"])
                exposure_metadata(work,frame)
        inputs.append("exposures.csv")
    with run_stage(work,args.config,args.region,stage,inputs=inputs,outputs=outputs) as go:
        if go:
            footprint_cache=None if args.preflight is None else args.preflight/"cache"
            compute(work,args.region,spec,config,settings,custom,args.legacy,footprint_cache)


if __name__=="__main__":
    main()
