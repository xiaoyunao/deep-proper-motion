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
from .access import Cache
from .astrometry import project, unproject
from .dark_precision import query_unique
from .exposure_catalogues import inside
from .exposure_model import hierarchy, observation_keys
from .io import save_json
from .overlap_calibration import assess_boundary, overlap_trial
from .nsc_comparison import comparison, qso_matches
from .paired_control import catalogue_motion, common_catalogue_ids
from .pilot_production import make_tracks, quality_detections, source_funnel, valid_epochs
from .pilot_validation import track_audit
from .precision_v03 import calibration, fit_with_calibration
from .provenance import run_stage
from .regional_validation import official_table, tag_magnitude, valid_published
from .validation import gaia_labels, groups_by_label
from .validation_audit import (anchored_bright_gaia, boundary_pairs, dark_quality, gaia_bright, gaia_observation_quality,
                               graph_support, held_conditions, sdss_field_overlap, sdss_overlap_rows)


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


def held_galaxies(refs,data,xy,nsc,keys,offsets,pixels,fine,settings):
    names = ("C0", "C1", "C1b")[:len(offsets)]
    a = keys[0]
    use = refs.is_holdout.copy()
    for key, model in zip(keys, offsets):
        use &= np.isin(key[refs.index], list(model))
    rows=refs.loc[use]
    points=[]
    boundary=defaultdict(list)
    diagnostics=defaultdict(list)
    color=data.color.to_numpy()
    magnitude=data.magnitude.to_numpy()
    band=data.band.to_numpy()
    airmass=data.airmass.to_numpy()
    fwhm=data.fwhm.to_numpy()
    mag_error=data.magnitude_error.to_numpy()
    centre=np.median(xy,axis=0)
    for _,row in rows.groupby("reference",sort=False):
        i=row.index.to_numpy()
        early,late=i[~nsc[i]],i[nsc[i]]
        if not len(early) or not len(late):
            continue
        corrected = [xy[i]-np.stack([model[key[j]][0] for j in i]) for key, model in zip(keys, offsets)]
        values = np.array([np.median(x[~nsc[i]], axis=0)-np.median(x[nsc[i]], axis=0) for x in corrected])
        points.append({name: (v, np.ones(2)) for name, v in zip(names, values)})
        point=np.median(xy[late],axis=0)-centre
        terms=["sky_"+("east" if point[0]>0 else "west")+("_north" if point[1]>0 else "_south")]
        med=np.median(magnitude[early])
        terms.append("r_20_to_22" if med>=20 else "r_below_20")
        nscr=late[band[late]=="r"]
        if np.isfinite(fwhm[nscr]).any():
            seeing=np.nanmedian(fwhm[nscr])
            terms.append("nsc_r_fwhm_ge_1_3" if seeing>=1.3 else "nsc_r_fwhm_lt_1_3")
        if np.isfinite(mag_error[nscr]).any():
            error=np.nanmedian(mag_error[nscr])
            terms.append("nsc_r_magerr_ge_0_05" if error>=.05 else "nsc_r_magerr_lt_0_05")
        c=color[early]
        if np.isfinite(c).any():
            c=np.median(c[np.isfinite(c)])
            terms.append("g_i_red" if c>=1.2 else "g_i_blue" if c<.5 else "g_i_mid")
        for term in terms:
            diagnostics[term].append(values)
        for passband in set(band[late])-{"r"}:
            same=band[i]==passband
            other=(band[i]=="r")&nsc[i]
            if other.any():
                delta=[np.median(x[same],axis=0)-np.median(x[other],axis=0) for x in corrected]
                diagnostics["nsc_r_to_"+passband].append(delta)
        # A matched galaxy seen by different detectors of the same parent probes an actual edge.
        for group in set(a[i]):
            q=np.flatnonzero(a[i]==group)
            if len(set(fine[i[q]]))>1:
                boundary["physical_detector"].append([
                    np.max(np.linalg.norm(x[q]-np.median(x[q],axis=0),axis=1)) for x in corrected])
        if len(set(pixels[i]))>1:
            boundary["cross_pixel_position"].append(np.linalg.norm(values,axis=1))
    stats=comparison(points,dict(settings,tail_masyr=150),(("C1","C0"),("C1b","C0"),("C1b","C1")))
    for result in stats.get("methods",{}).values():
        for key in ("standardized_nmad","fraction_within_1sigma","fraction_within_2sigma"):
            del result[key]
    return dict(comparison=stats,
                boundary={k:dict(n=len(v),p68_mas=np.quantile(v,.68,axis=0)) for k,v in boundary.items()},
                residual_strata={k:dict(n=len(v),median_mas=np.median(v,axis=0),
                       radial_p68_mas=np.quantile(np.linalg.norm(v,axis=2),.68,axis=0))
                       for k,v in diagnostics.items() if len(v)>=10},
                dcr_metadata=dict(held_reference_measurements=len(rows),
                                  finite_airmass=int(np.isfinite(airmass[rows.index]).sum()),
                                  nsc_airmass=int(np.count_nonzero(nsc[rows.index]&np.isfinite(airmass[rows.index])))))


def pilot_gate(galaxy,qso,settings):
    galaxy_fit=galaxy["comparison"]
    qso_fit=qso["by_cohort"].get("c0_c1",{}).get("strict__all__all",{})
    galaxies=galaxy_fit.get("paired",{}).get("C1_vs_C0",{})
    quasars=qso_fit.get("paired",{}).get("C1_vs_C0",{})
    gal_gain=galaxies.get("fraction_improved",[None]*3)[2]
    gal_ci=galaxies.get("fraction95",[None]*3)[2]
    qso_ci=quasars.get("fraction95",[None]*3)[2]
    adequate=qso_fit.get("n",0)>=settings["min_qso_count"]
    improvement=gal_gain is not None and gal_gain>=settings["min_galaxy_gain"] and gal_ci[0]>0
    no_loss=qso_ci is not None and qso_ci[0]>=-settings["max_qso_relative_loss"]
    return dict(galaxy_position_gain=bool(improvement),qso_sample=adequate,qso_noninferiority=bool(no_loss),
                primary_metrics_pass=bool(improvement and adequate and no_loss))


def contrast_samples(truth, labels, data, xy, t, sigma, keep, nsc, settings, config,
                     keys, offsets, points, magnitudes, origin, ref, ref_error,
                     release, legacy, tracks, products, block_records=None):
    truth={k:i for k,i in truth.items() if nsc[i].any() and not nsc[i].all()}
    linked,status=common_catalogue_ids(truth,labels,data,release[0])
    published=release[1]
    audits={k:track_audit(tracks,values,truth,labels,nsc,
                          settings["production"]["validation_min_recovery"]) for k,values in products.items()}
    names = ("C0", "C1", "C1b")[:len(offsets)]
    support = np.logical_and.reduce([np.isin(k, list(m)) for k, m in zip(keys, offsets)])
    paired_key = "c0_c1" if len(names) == 2 else "c0_c1_c1b"
    full_key = "all_four" if len(names) == 2 else "all_five"
    full_count = len(names) + 2
    records=defaultdict(lambda:defaultdict(list))
    temporal=defaultdict(list)
    totals=Counter()
    exposure=data.exposure_id.to_numpy()
    for k,detections in truth.items():
        if k>=len(origin) or not np.isfinite(magnitudes[k]):
            continue
        indices=detections[support[detections]&keep[detections]]
        if not valid_epochs(indices,nsc,t,settings["production"]):
            continue
        base={}
        for name,keyset,solution in zip(names,keys,offsets):
            fit,err=fit_with_calibration(xy[indices],t[indices],sigma[indices],config["reference_epoch"],
                                        [solution[keyset[i]] for i in indices])
            base[name]=(fit.coeff[1],err)
        if linked.get(k) in published.index:
            values=valid_published(published.loc[linked[k]],points)
            if values is not None:
                base["NSC"]=values
        if k in legacy:
            base["Xiao"]=legacy[k]
        if block_records is not None and 21<=magnitudes[k]<23:
            block_records.append(dict(magnitude=magnitudes[k],
                exposures=sorted(set(exposure[indices])),
                nsc_epoch_days=sorted(set(np.floor((t[indices[nsc[indices]]]-2000)*365.25).astype(int))),
                motion={name:pm-ref[k] for name,(pm,_) in base.items()}))
        blind={name:products[name][found[k]][1:3] for name,(found,_) in audits.items() if k in found}
        if all(name in blind for name in names):
            blind.update({name:x for name,x in base.items() if name in ("Xiao","NSC")})
        else:
            blind={}
        totals["strict"]+=1
        totals["strict_all_catalogues"]+=len(base)==full_count
        totals["blind_both_models"]+=bool(blind)
        totals["blind_all_catalogues"]+=len(blind)==full_count
        for state,values in (("strict",base),("blind",blind)):
            if not values:
                continue
            corrected={name:(pm-ref[k],np.hypot(error,ref_error[k])) for name,(pm,error) in values.items()}
            for mag in tag_magnitude(magnitudes[k],settings):
                if state=="strict":
                    late=nsc[indices]
                    temporal[mag].append([np.count_nonzero(~late),len(set(exposure[indices[~late]])),
                                          len(set(exposure[indices[late]])),np.ptp(t[indices])])
                for source in {"all",origin[k]}:
                    prefix=f"{state}__{source}__{mag}"
                    records[paired_key][prefix].append({name: corrected[name] for name in names})
                    if len(values)==full_count:
                        records[full_key][prefix].append(corrected)
    comparisons=(("C1","C0"),("C1","NSC"),("C1","Xiao"),("C0","Xiao"),("C0","NSC"),
                 ("C1b","C0"),("C1b","C1"),("C1b","NSC"),("C1b","Xiao"))
    cohorts={kind:{label:comparison(r,settings,comparisons) for label,r in bins.items()}
             for kind,bins in records.items()}
    for values in cohorts.values():
        for cohort in values.values():
            for stats in cohort.get("methods",{}).values():
                lower,upper=np.asarray(stats["median_bias95"]).T
                limit=settings["zero_point_limit_masyr"]
                stats["zero_by_axis"]=["within" if lo>=-limit and hi<=limit else
                    "outside" if lo>limit or hi<-limit else "unresolved" for lo,hi in zip(lower,upper)]
    coverage={k:dict(n=len(v),p16_median_p84=np.quantile(v,[.16,.5,.84],axis=0))
              for k,v in temporal.items() if len(v)>=settings["min_publish_count"]}
    return dict(totals=totals,archive_match_status=status,by_cohort=cohorts,temporal=coverage,
                blind_quality={name:audit for name,(_,audit) in audits.items()})


def run(work,inputs,legacy_path,spec,calibration_config,audit=False,sensitivity=False,sas=None,bright_only=False,overlap_options=None):
    start=monotonic()
    settings=tomllib.loads(Path("configs/spatial-v08.toml").read_text())
    settings["bootstrap_draws"]=400
    settings["bootstrap_absolute_zero"]=True
    config=tomllib.loads(Path("configs/precision-v03.toml").read_text())
    center=tuple(np.mean(spec[k]) for k in ("ra","dec"))
    radius=1.5
    data,xy,t,sigma,keep=quality_detections(inputs,center,config,settings["production"])
    nsc=data.survey.eq("NSC").to_numpy()
    print(f"{len(data):,} normalized positions with usable errors, {np.count_nonzero(keep):,} r detections",flush=True)
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
    overlap_summary = {}
    if overlap_options is not None:
        edges, training = overlap_trial(sas, refs, data, xy, sigma, pairs, models, center, overlap_options)
        new_keys, c1b, c1b_stats = hierarchy(data, refs, xy, sigma, config, calibration_config, c0, overlaps=edges)
        held, counts = sdss_overlap_rows(sas, refs, data, xy, pairs, models, center, holdout=True)
        boundary = assess_boundary(held, counts, (*models, c1b), overlap_options, config["seed"])
        overlap_summary = dict(overlap_training=training, overlap_boundary=boundary, c1b=c1b_stats)
        pairs = (*pairs, new_keys)
        models = (*models, c1b)
        print(f"C1b: {training}, {c1b_stats}", flush=True)
    names = ("C0", "C1", "C1b")[:len(models)]
    galaxy=held_galaxies(refs,data,xy,nsc,pairs,models,pixels,fine,settings)
    print(f"C1: {c1_stats}; held galaxies {galaxy['comparison']['n']}",flush=True)
    visits=data.exposure_id.to_numpy()
    tracks=make_tracks(xy,visits,keep,1000.)
    seeds=make_tracks(xy,visits,keep&nsc,1000.)
    print(f"{len(tracks):,} shared blind associations, {len(seeds):,} NSC anchors",flush=True)
    products={}
    yields={}
    quality={}
    def core(position):
        ra,dec=unproject(position,center)[0]
        return inside(ra,dec,spec)
    for name,obs,solution in zip(names,pairs,models):
        failed=[] if audit else None
        yields[name],products[name]=source_funnel(data,xy,t,sigma,keep,tracks,solution,config,settings,radius,
            initial=seeds,observation_groups=obs,core_members=core,quality_audit=failed)
        if audit:
            quality[name]=dark_quality(failed)
        print(f"{name}: {len(products[name]):,} blind quality solutions",flush=True)
    original=pd.read_csv(inputs/"external_labels.csv",dtype=str).set_index("detection_id").original_object_id
    release=(original,official_table(inputs,{}))
    legacy=pd.read_csv(legacy_path)
    legacy=legacy.loc[inside(legacy.RA,legacy.DEC,spec,.06)]
    if not bright_only:
        projected=project(qso.ra,qso.dec,center)
        labels,matches=qso_matches(data,xy,keep,projected)
        old,old_count=matched_legacy(legacy,projected,center)
        block_records=[] if sas else None
        qso_stats=contrast_samples(matches,labels,data,xy,t,sigma,keep,nsc,settings,config,
            pairs,models,center,qso.reference_r.to_numpy(),qso.origin.to_numpy(),
            np.zeros((len(qso),2)),np.zeros((len(qso),2)),release,old,tracks,products,block_records)
        if sas:
            save_json(work/"qso_observation_blocks.json",block_records)
        print(f"QSO controls: {qso_stats['totals']}",flush=True)
    g=pd.read_csv(inputs/"validation/gaia.csv")
    if audit and "visibility_periods_used" not in g.columns:
        visibility=gaia_observation_quality(Cache(work/"gaia_quality_cache"),spec)
        g=g.merge(visibility,on="source_id",how="left",validate="one_to_one")
    gaia_audit={}
    if audit:
        g,gaia_audit=anchored_bright_gaia(g,seeds,xy,data.magnitude.to_numpy(),center)
    labels0,cat=gaia_labels(data.loc[keep].reset_index(drop=True),xy[keep],t[keep],sigma[keep],
                           g,center,dict(config,gaia_max_pm_error=2.))
    labels=np.full(len(data),-1)
    labels[keep]=labels0
    truth=groups_by_label(labels,data,t,cat,center,radius,config)
    truth={i:idx for i,idx in truth.items() if inside(cat.ra.iloc[i],cat.dec.iloc[i],spec)}
    mag=np.full(len(cat),np.nan)
    for i,idx in truth.items():
        mag[i]=np.median(data.magnitude.iloc[idx[nsc[idx]]])
    if audit:
        bright=gaia_bright(cat,mag)
        gaia_audit.update(cross_survey_labels=len(truth),bright_labels=int(bright.sum()),
                          excluded_final_faint_labels=int(np.count_nonzero(mag>=21)))
        truth={i:idx for i,idx in truth.items() if bright[i]}
        labels[~np.isin(labels,list(truth))]=-1
    old_gaia,old_gaia_counts=matched_legacy(legacy,project(cat.ra,cat.dec,center),center)
    gaia_stats=contrast_samples(truth,labels,data,xy,t,sigma,keep,nsc,settings,config,
        pairs,models,center,mag,np.repeat("Gaia",len(cat)),cat[["pm_tangent_x","pm_tangent_y"]].to_numpy(),
        cat[["pmra_error","pmdec_error"]].to_numpy(),release,old_gaia,tracks,products)
    if bright_only:
        save_json(work/"bright_summary.json",dict(gaia_policy=gaia_audit,gaia=gaia_stats))
        return
    extras={}
    if audit:
        print(f"Bright Gaia: {gaia_stats['totals']}",flush=True)
        extras=dict(gaia_policy=gaia_audit, dark_quality=quality,
                    reference_graph=graph_support(refs,pairs[1],c1),
                    physical_boundaries=boundary_pairs(refs,data,xy,data.detection_id.map(original),pairs[:2],models[:2],center,spec))
        if sas:
            extras["physical_boundaries"]["sdss_adjacent_field"] = sdss_field_overlap(sas,refs,data,xy,pairs,models,center,spec)
            galaxy["held_observing_conditions"] = held_conditions(refs,data,xy,keys,c1)
        extras["sensitivity"]={}
        if sensitivity:
            for references,prior in ((10,50.),(25,50.),(15,25.),(15,100.)):
                alternative=dict(calibration_config,min_child_references=references,child_prior_mas=prior)
                altkeys,fit,diagnostic=hierarchy(data,refs,xy,sigma,config,alternative,c0)
                held=held_galaxies(refs,data,xy,nsc,(pairs[0],altkeys),(c0,fit),pixels,fine,settings)
                extras["sensitivity"][f"references_{references}_prior_{int(prior)}"]=dict(support=diagnostic,
                                                                            matched_galaxies=held["comparison"])
                print(f"Sensitivity {references} / {prior} complete",flush=True)
        from .regional_production import prediction
        multi=keep|(nsc&data.band.isin(["g","i","z"]).to_numpy()&data.magnitude_error.between(0,.21715).to_numpy())
        shapes=dict(spec,radius=radius,core_members=core)
        extras["annual_holdout"]=prediction(work,center,data,xy,t,sigma,sigma,keys,nsc,keep,multi,c1,
                                             settings,config,shapes)
        if overlap_options is not None:
            extras["annual_holdout_c1b"] = prediction(
                work,center,data,xy,t,sigma,sigma,new_keys,nsc,keep,multi,c1b,settings,config,shapes)
    save_json(work/"pilot_summary.json",dict(c0=c0_stats,c1=c1_stats,galaxies=galaxy,qso=qso_stats,
        gaia=gaia_stats,legacy_origin_counts=dict(qso=old_count,gaia=old_gaia_counts),source_yield=yields,
        gate=pilot_gate(galaxy,qso_stats,calibration_config),
        qso_origins=qso.origin.value_counts().to_dict(),elapsed_seconds=monotonic()-start,
        acquisition=json.loads((inputs/"acquisition_summary.json").read_text()),**extras,**overlap_summary))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("root",type=Path)
    p.add_argument("region")
    p.add_argument("inputs",type=Path)
    p.add_argument("legacy",type=Path)
    p.add_argument("config",type=Path)
    p.add_argument("--validation-audit",action="store_true")
    p.add_argument("--sensitivity",action="store_true")
    p.add_argument("--sdss-sas",type=Path)
    p.add_argument("--bright-only",action="store_true")
    a=p.parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        p.error("Run on a Slurm compute node")
    settings=tomllib.loads(a.config.read_text())
    work=a.root/a.region
    work.mkdir(parents=True,exist_ok=True)
    input_receipt=(a.inputs/a.region/"run_manifest.json").resolve(strict=True)
    if not (work/"input_manifest.json").exists():
        (work/"input_manifest.json").symlink_to(input_receipt)
    if (work/"input_manifest.json").resolve()!=input_receipt:
        raise ValueError("Pilot is already bound to a different acquisition")
    if a.bright_only and (not a.validation_audit or a.sdss_sas):
        p.error("Bright-only replay requires validation audit with cached measurements")
    overlap = settings.get("overlap")
    if overlap is not None and not (a.validation_audit and a.sdss_sas and not a.bright_only):
        p.error("Overlap experiment requires --validation-audit and --sdss-sas")
    stage="overlap-pilot-v11" if overlap is not None else "validation-bright-v10" if a.bright_only else "exposure-pilot-v10" if a.validation_audit else "exposure-pilot-v09"
    outputs=["bright_summary.json" if a.bright_only else "pilot_summary.json"]
    if a.sdss_sas:
        outputs.append("qso_observation_blocks.json")
    with run_stage(work,a.config,a.region,stage,inputs=["input_manifest.json"],outputs=outputs) as go:
        if go:
            run(work,a.inputs/a.region,a.legacy,settings["fields"][a.region],settings["calibration"],
                a.validation_audit,a.sensitivity,a.sdss_sas,a.bright_only,overlap)


if __name__=="__main__":
    main()
