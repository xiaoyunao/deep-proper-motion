"""Evaluate v0.6 astrometric alternatives on frozen real SDSS and NSC detections."""
import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
from time import monotonic, process_time

from astropy.coordinates import get_body_barycentric
from astropy.time import Time
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from . import access
from .__main__ import tomllib
from .astrometry import project, seasons
from .astrometry_weights import correlated_fit, parallax_factors, variance_model
from .dark_precision import bin_name, query_unique
from .epoch_statistics import footprint, observation_groups, source_coverage
from .io import save_json
from .nsc_comparison import comparison, configuration, qso_matches
from .paired_control import catalogue_motion, common_catalogue_ids, published_rows
from .pilot_production import make_tracks, quality_detections, source_funnel, timing_resources, valid_epochs
from .pilot_validation import track_audit
from .precision_v03 import calibration, calibration_groups, fit_with_calibration
from .provenance import run_stage
from .validation import gaia_labels, groups_by_label, ratio


def qtile(values):
    return np.quantile(values, [.16,.50,.84], axis=0) if len(values) >= 3 else None


def exposure_metadata(work, data):
    exposure = data.loc[data.survey.eq("NSC"), "exposure_id"].str[len("NSC:"):].unique()
    cache = access.Cache(work/"metadata_cache")
    blocks = []
    for start in range(0, len(exposure), 120):
        values = ",".join(map(access.literal, exposure[start:start+120]))
        blocks.append(cache.checked("nsc", "exposure,instrument,airmass", "nsc_dr2.exposure",
                                    f"exposure IN ({values})"))
    meta = pd.concat(blocks, ignore_index=True).drop_duplicates()
    if meta.exposure.duplicated().any():
        raise ValueError("Conflicting NSC exposure metadata")
    meta.to_csv(work/"exposures.csv", index=False)


def quality_flags(data):
    flags = data.quality_flags.fillna(0).to_numpy(dtype=np.int64)
    nsc = data.survey.eq("NSC").to_numpy()
    # SExtractor saturation/incomplete/truncation/overflow; SDSS EDGE/SATURATED.
    clear = np.where(nsc, (flags & 252) == 0, (flags & (0x4 | 0x40000)) == 0)
    return clear & data.quality_flags.notna().to_numpy() & (nsc | ~data.nchild.gt(0).to_numpy())


def calibrate_errors(data, xy, raw, corrected, refs, features, groups, support, settings):
    pairs = defaultdict(list)
    residual = np.full_like(xy, np.nan)
    train = refs.loc[~refs.is_holdout & support[refs.index]]
    held = refs.loc[refs.is_holdout & support[refs.index]]
    for _, rows in train.groupby("reference",sort=False):
        for key, frame in rows.groupby(features[rows.index],sort=False):
            ix = frame.sort_values("observation_time").index.to_numpy()
            if len(set(groups[ix])) >= 2:
                residual[ix] = corrected[ix] - np.median(corrected[ix],axis=0)
            # One representative per night/run prevents dense sequences dominating.
            ix = ix[~pd.Series(groups[ix]).duplicated().to_numpy()]
            left, right = ix[::2], ix[1::2]
            pairs[key].extend(zip(left[:len(right)],right))
    errors = np.hypot(raw,50)
    shared = np.zeros_like(errors)
    summary = {}
    for key in sorted(set(features)):
        ii = features==key
        contrast = np.asarray(pairs[key],dtype=int).reshape(-1,2)
        left, right = contrast.T
        coeff, trained = [], []
        for axis in range(2):
            pair, status = variance_model(corrected[left,axis]-corrected[right,axis],
                                             raw[left,axis],raw[right,axis],settings["minimum_pairs"])
            coeff.append(pair)
            trained.append(status)
        scale, floor = np.array(coeff).T
        errors[ii] = np.hypot(raw[ii]*scale,floor)
        summary[key] = dict(detections=int(ii.sum()), finite_seeing=int(data.fwhm.iloc[ii].notna().sum()),
                            raw_mas=qtile(raw[ii]), W0_mas=qtile(np.hypot(raw[ii],50)),
                            W1_mas=qtile(errors[ii]), fit_pairs=len(contrast),
                            scale=scale, floor_mas=floor, calibrated_axes=trained)
    # Estimate remaining shared group variance from training group medians.
    # W1 measures marginal variance; W2 partitions it instead of adding it again.
    survey = data.survey.to_numpy()
    for family in ("SDSS","NSC"):
        ii = (survey==family) & np.isfinite(residual).all(axis=1)
        frame = pd.DataFrame(residual[ii],columns=["x","y"]).assign(group=groups[ii])
        binned = frame.groupby("group")
        med = binned.median().loc[binned.size() >= 10]
        common = 1.4826*np.median(abs(med-med.median()),axis=0) if len(med)>=5 else np.zeros(2)
        for key,item in summary.items():
            if key.startswith(family+"_"):
                use = features==key
                allocated = np.minimum(common,.8*np.asarray(item["floor_mas"]))
                shared[use] = allocated
                item.update(shared_groups=len(med), shared_mas=allocated)
    checks = defaultdict(list)
    for _, rows in held.groupby("reference",sort=False):
        for key, frame in rows.groupby(features[rows.index],sort=False):
            ix = frame.sort_values("observation_time").index.to_numpy()
            ix = ix[~pd.Series(groups[ix]).duplicated().to_numpy()]
            left,right = ix[::2],ix[1::2]
            left = left[:len(right)]
            for a,b in zip(left,right):
                d = abs(corrected[a]-corrected[b])
                checks[(key,"all")].append((d/np.hypot(np.hypot(raw[a],50),np.hypot(raw[b],50)),
                                           d/np.hypot(errors[a],errors[b])))
                checks[(key,"bright" if np.mean(data.magnitude.iloc[[a,b]])<21 else "faint")].append(
                    checks[(key,"all")][-1])
                snr = np.mean(data.magnitude_error.iloc[[a,b]])
                seeing = np.mean(data.fwhm.iloc[[a,b]])
                if np.isfinite(snr):
                    checks[(key,"snr_ge_15" if snr<2.5/(15*np.log(10)) else "snr_lt_15")].append(checks[(key,"all")][-1])
                if np.isfinite(seeing):
                    checks[(key,"fwhm_lt_1p3" if seeing<1.3 else "fwhm_ge_1p3")].append(checks[(key,"all")][-1])
    for (key,tag), values in checks.items():
        arr = np.asarray(values)
        summary[key].setdefault("held_galaxies",{})[tag] = dict(n=len(arr),
            normalized_abs68=np.quantile(arr,.68,axis=0) if len(arr)>=3 else None,
            within_1sigma=np.mean(arr<=1,axis=0) if len(arr)>=3 else None)
    return errors, shared, summary


def augment_tracks(tracks, data, xy, eligible, r_keep, prod):
    if not tracks:
        return []
    anchors = np.array([np.median(xy[index],axis=0) for index in tracks])
    tree = cKDTree(anchors)
    extra = defaultdict(list)
    for _, frame in data.loc[eligible & ~r_keep].groupby("exposure_id",sort=False):
        j = frame.index.to_numpy()
        match, winner = query_unique(tree,xy[j],prod["match_radius_mas"],prod["competitor_margin_mas"])
        # query_unique applies exclusivity among the detections of this exposure.
        for source,detection in zip(winner[match],j[match]):
            extra[source].append(detection)
    return [np.r_[index,np.asarray(extra[j],dtype=int)] for j,index in enumerate(tracks)]


def make_fit(xy,t,sigma,keys,shared,epoch,samples):
    return correlated_fit(xy,t,np.sqrt(sigma**2-shared**2),keys,shared,epoch,samples)


def group_checks(xy,t,sigma,keys,shared,nsc,quality,epoch,samples,base,settings,prod):
    index = np.arange(len(t))
    quality_use = index[quality]
    if not valid_epochs(quality_use,nsc,t,prod):
        return None, dict(no_fit=1, n=len(t), bad_flags=int((~quality).sum()))
    model = make_fit(xy[quality_use],t[quality_use],sigma[quality_use],keys[quality_use],
                     shared[quality_use],epoch,samples[:,quality_use])
    importance = np.linalg.norm(base.influence[[1,3]],axis=(0,2))
    share = {key:importance[keys==key].sum()/importance.sum() for key in set(keys)}
    selected = sorted(set(keys[quality_use]),key=lambda key:(-share[key],key))
    shifts = []
    removable = []
    for key in selected:
        left = index[(keys!=key) & quality]
        if not valid_epochs(left,nsc,t,prod):
            continue
        reduced = make_fit(xy[left],t[left],sigma[left],keys[left],shared[left],epoch,samples[:,left])
        shift = np.linalg.norm(reduced.motion-model.motion)
        shifts.append(shift)
        remain = index[(keys==key) & quality]
        dt = t[remain]-epoch
        predicted = reduced.params[[0,2]]+dt[:,None]*reduced.motion
        variance = sigma[remain]**2+np.column_stack([
            reduced.covariance[a,a]+2*dt*reduced.covariance[a,a+1]+dt**2*reduced.covariance[a+1,a+1]
            for a in (0,2)])
        if (np.linalg.norm((xy[remain]-predicted)/np.sqrt(variance),axis=1)
                >settings["robust_remove_sigma"]).all():
            removable.append((shift,key))
        if len(shifts)>=2:
            break
    removed = 0
    if removable:
        cut = max(removable)[1]
        used = quality_use[keys[quality_use]!=cut]
        removed = len(quality_use)-len(used)
        model = make_fit(xy[used],t[used],sigma[used],keys[used],shared[used],epoch,samples[:,used])
    return model, dict(n=len(t), bad_flags=int((~quality).sum()), removed=removed,
                       downweighted=int((base.weights<.5).sum()), max_group_fraction=max(share.values()),
                       max_tested_shift=max(shifts) if shifts else None, groups=len(share),
                       nights_or_runs_kept=len(set(keys[quality_use]))-bool(removed),
                       reduced_chi2=base.chi2/(2*len(t)-4) if len(t)>2 else None,
                       unique_sdss_run=len(set(keys[~nsc]))==1)


def fit_controls(work, center, spec, data, xy, t, sigma, r_keep, multi_keep, offsets, arrays,
                 keep_tracks, selected, config, settings):
    keys, groups, corrected, support, nsc, w1, common = arrays
    prod = settings["production"]
    epoch = config["reference_epoch"]
    flags = quality_flags(data)
    all_qso = pd.read_csv(work/"spectroscopic_qso.csv")
    qso = all_qso.loc[all_qso.in_core].reset_index(drop=True)
    qxy = project(qso.ra,qso.dec,center)
    labels, q_truth = qso_matches(data,xy,r_keep,qxy)
    _, q_multi = qso_matches(data,xy,multi_keep,qxy)
    q_blind, _ = track_audit(keep_tracks,selected,q_truth,labels,nsc,prod["validation_min_recovery"])
    raw = pd.read_csv(work/"external_labels.csv",dtype=str).set_index("detection_id").original_object_id
    linked, _ = common_catalogue_ids({k:v for k,v in q_truth.items() if nsc[v].any()},labels,data,raw)
    published = published_rows(access.Cache(work/"validation_cache"),linked.values())
    gaia = pd.read_csv(work/"validation"/"gaia.csv")
    # Near-star matching computes annual parallax only in this post-association evaluation.
    idx = np.flatnonzero(multi_keep & support)
    subset, gaia = gaia_labels(data.iloc[idx].reset_index(drop=True),xy[idx],t[idx],sigma[idx],gaia,center,
                               dict(config,truth_mode="annual_parallax",gaia_max_pm_error=2.))
    gaia_label = np.full(len(data),-1)
    gaia_label[idx] = subset
    g_multi = groups_by_label(gaia_label,data,t,gaia,center,spec["radius"],config)
    g_truth = {k:v[r_keep[v]] for k,v in g_multi.items()}
    mags = data.magnitude.to_numpy()
    case_rng = np.random.default_rng(settings["seed"])
    g_eligible = [i for i in g_truth if valid_epochs(g_truth[i],nsc,t,prod)]
    g_near = [i for i in g_eligible if gaia.parallax.iloc[i] >= 10]
    g_other = [i for i in g_eligible if gaia.parallax.iloc[i] < 10]
    g_ids = g_near+sorted(case_rng.choice(g_other,min(len(g_other),settings["max_evaluation_gaia"]),replace=False))
    unique, inverse = np.unique(t,return_inverse=True)
    observer = get_body_barycentric("earth",Time(unique,format="jyear",scale="tt")).xyz.to_value("AU").T
    categories, influence, physical, excl = defaultdict(list), defaultdict(list), defaultdict(list), Counter()
    timing = defaultdict(list)
    survey = data.survey.to_numpy()

    def pm_result(model, ref, ref_error):
        return model.motion-ref, np.hypot(model.error,ref_error)

    for origin, selected_ids, truth_map, expanded in (
            ("qso", sorted(q_truth),q_truth,q_multi),("gaia",g_ids,g_truth,g_multi)):
        for i in selected_ids:
            index = truth_map[i]
            index = index[support[index]]
            if not valid_epochs(index,nsc,t,prod):
                excl[origin+"_no_frozen_m3"] += 1
                continue
            augmented = expanded[i]
            augmented = np.union1d(index,augmented[multi_keep[augmented] & support[augmented]])
            if origin=="qso":
                key = linked.get(i)
                if key not in published.index:
                    excl["qso_missing_published"] += 1
                    continue
                row = published.loc[key]
                official = (catalogue_motion(row,center),row[["pmraerr","pmdecerr"]].to_numpy(dtype=float))
                if (not np.isfinite(np.r_[official[0],official[1],row.deltamjd]).all()
                        or np.any(official[1]<=0) or np.any(abs(official[0])>1e5) or row.deltamjd<=0):
                    excl["qso_invalid_published"] += 1
                    continue
                ref,err = np.zeros(2),np.zeros(2)
                r = qso.reference_r.iloc[i]
            else:
                p = gaia.iloc[i]
                ref = p[["pm_tangent_x","pm_tangent_y"]].to_numpy(dtype=float)
                err = p[["pmra_error","pmdec_error"]].to_numpy(dtype=float)
                r = np.median(mags[index[nsc[index]]])
            offsets_here = [offsets[k] for k in keys[index]]
            draws = np.stack([item[1] for item in offsets_here],axis=1)
            base,error = fit_with_calibration(xy[index],t[index],sigma[index],epoch,offsets_here)
            fitted = {"W0":(base.coeff[1]-ref,np.hypot(error,err))}
            if origin=="qso":
                fitted["published_nsc"] = official
            for name,corr in (("W1",np.zeros_like(common[index])),("W2",common[index])):
                model = make_fit(corrected[index],t[index],w1[index],groups[index],corr,epoch,draws)
                fitted[name] = pm_result(model,ref,err)
            if len(augmented)>len(index):
                a = augmented
                draws_multi = np.stack([offsets[key][1] for key in keys[a]],axis=1)
                fit_multi = make_fit(corrected[a],t[a],w1[a],groups[a],np.zeros_like(w1[a]),epoch,draws_multi)
                both_bands = dict(W1=fitted["W1"],W1_multi=pm_result(fit_multi,ref,err))
                categories[origin+"_added_bands"].append(both_bands)
                categories[origin+"_added_bands_"+bin_name(r,settings["reference_r_edges"])].append(both_bands)
            tags = [origin+"_all",origin+"_"+bin_name(r,settings["reference_r_edges"])]
            if origin=="qso" and i in q_blind:
                j = q_blind[i]
                _, old_pm, old_error, _, b, _ = selected[j]
                bdraws = np.stack([offsets[key][1] for key in keys[b]],axis=1)
                blind = dict(W0=(old_pm,old_error),published_nsc=official)
                for name,corr in (("W1",np.zeros_like(common[b])),("W2",common[b])):
                    bf = make_fit(corrected[b],t[b],w1[b],groups[b],corr,epoch,bdraws)
                    blind[name] = pm_result(bf,ref,err)
                categories["qso_blind_common"].append(blind)
                if 22<=r<23:
                    categories["qso_blind_22_to_23"].append(blind)
            early = ~nsc[index]
            if origin=="qso" or i in g_near or len(influence[origin+"_all"]) < 500:
                check, info = group_checks(corrected[index],t[index],w1[index],groups[index],common[index],
                                           nsc[index],flags[index],epoch,draws,model,settings,prod)
                if check is not None:
                    quality_fit = dict(fitted,Q=pm_result(check,ref,err))
                    categories[origin+"_quality"].append(quality_fit)
                    if 22<=r<23:
                        categories[origin+"_quality_22_to_23"].append(quality_fit)
                for group in (origin+"_all","one_sdss" if len(set(groups[index][early]))==1 else "repeat_sdss",
                              "gt_10_sdss" if len(set(groups[index][early]))>10 else None):
                    if group:
                        influence[group].append(info)
            for tag in tags:
                categories[tag].append(fitted)
            # Report fractional slope influence using its L1 norm per survey.
            leverage = np.linalg.norm(model.influence[[1,3]],axis=(0,2))
            timing[origin].append((early.sum(),np.ptp(t[index]),leverage[early].sum()/leverage.sum(),
                                   np.sum(w1[index][early]**-2)/np.sum(w1[index]**-2)))
            if origin=="gaia":
                distance = "near" if p.parallax>=10 else "far" if abs(p.parallax)<2 else "intermediate"
                if distance=="intermediate":
                    continue
                physical[distance].append(None)
                if (len(set(groups[index]))<settings["minimum_parallax_groups"] or
                        np.ptp(t[index][~early])<settings["minimum_parallax_nsc_span_years"] or
                        len(set(seasons(t[index][~early],75)))<3):
                    continue
                factors = parallax_factors(observer[inverse[index]],center,np.median(xy[index],axis=0))
                try:
                    five = correlated_fit(corrected[index],t[index],np.sqrt(w1[index]**2-common[index]**2),
                                          groups[index],common[index],epoch,draws,factors)
                except np.linalg.LinAlgError:
                    continue
                phase = np.sqrt(five.parallax_information)*np.median(w1[index])
                if not np.isfinite(phase) or phase<settings["min_parallax_phase_information"]:
                    continue
                physical[distance][-1] = (dict(L4=fitted["W2"],L5=pm_result(five,ref,err)),
                    np.linalg.norm(model.motion-five.motion), five.params[-1]-p.parallax,
                    np.sqrt(five.covariance[-1,-1]),phase)
    optional = (("W1","W0"),("W2","W0"),("W2","W1"),("W0","published_nsc"),
                ("W1","published_nsc"),("W2","published_nsc"),("Q","W2"),("W1_multi","W1"),("L5","L4"))
    metrics = {k:comparison(v,settings,optional) for k,v in categories.items()}
    physics = {}
    for key,values in physical.items():
        fit = [item for item in values if item is not None]
        physics[key] = dict(identifiable=ratio(len(fit),len(values)),
                            performance=comparison([item[0] for item in fit],settings,optional),
                            pm_shift_parallax_error_formal_and_phase=qtile([item[1:] for item in fit]))
    diagnosis = {tag:dict(n=len(values), flags_rejected=sum(v["bad_flags"] for v in values),
                          detections=sum(v["n"] for v in values),
                          downweighted=sum(v.get("downweighted",0) for v in values),
                          removed=sum(v.get("removed",0) for v in values),
                          no_fit=sum(v.get("no_fit",0) for v in values),
                          effective_groups=qtile([v["nights_or_runs_kept"] for v in values if "nights_or_runs_kept" in v]),
                          max_leverage=qtile([v["max_group_fraction"] for v in values if "max_group_fraction" in v]),
                          max_tested_pm_shift=qtile([v["max_tested_shift"] for v in values if v.get("max_tested_shift") is not None]),
                          no_testable_group=sum(v.get("max_tested_shift") is None for v in values),
                          residual_chi2_per_dof=qtile([v["reduced_chi2"] for v in values if v.get("reduced_chi2") is not None]))
                 for tag,values in influence.items()}
    return dict(motion=metrics, fits=diagnosis, timing={k:qtile(v) for k,v in timing.items()},
                parallax=physics, exclusions=dict(excl), gaia_valid_r_eligible=len(g_eligible))


def run(work, region, config, settings, custom, spec, footprint_index, stripe):
    wall,cpu = monotonic(),process_time()
    center = spec["ra"],spec["dec"]
    prod = settings["production"]
    geometry = footprint(footprint_index,spec,custom["footprint_samples"],custom["seed"])
    print("SDSS physical scan geometry complete.",flush=True)
    data,xy,t,sigma,keep = quality_detections(work,center,config,prod)
    qso = pd.read_csv(work/"spectroscopic_qso.csv")
    offsets,calinfo,refs = calibration(work,center,config,project(qso.ra,qso.dec,center),return_references=True)
    print("Frozen M3 offsets complete.",flush=True)
    keys = calibration_groups(data)
    assigned = np.array([offsets[key][0] if key in offsets else np.full(2,np.nan) for key in keys])
    corrected = xy-assigned
    support = np.isfinite(assigned).all(axis=1)
    nsc = data.survey.eq("NSC").to_numpy()
    meta = pd.read_csv(work/"exposures.csv").set_index("exposure")
    instruments = data.exposure_id.str[4:].map(meta.instrument).fillna("unknown").str.lower()
    # Known archive instrument strings, pooled as unknown on unexpected values.
    token = instruments.str.extract(r"^(c4d|k4m|ksb|decam|mosaic3|90prime)$",expand=False).fillna("unknown")
    features = np.where(nsc,"NSC_"+token+"_"+data.band,"SDSS_r")
    groups,night = observation_groups(data)
    raw = data[["sigma_east_mas","sigma_north_mas"]].to_numpy()
    w1,common,errors = calibrate_errors(data,xy,raw,corrected,refs,features,groups,support,custom)
    calinfo.pop("sdss_run_camcol",None)
    exp = data.exposure_id.to_numpy()
    tracks = make_tracks(xy,exp,keep,prod["match_radius_mas"])
    r_base,base = source_funnel(data,xy,t,sigma,keep,tracks,offsets,config,settings,spec["radius"])
    verified_band = {k for k,v in errors.items() if k.startswith("NSC_") and all(v["calibrated_axes"])}
    multi = keep | (nsc & data.band.isin(custom["bands"]).to_numpy() & np.isin(features,list(verified_band))
                    & data.magnitude_error.between(0,2.5/(np.log(10)*prod["min_snr"]),inclusive="right").to_numpy()
                    & data.magnitude.notna().to_numpy() & quality_flags(data))
    comparison_funnel, r_result = source_funnel(data,xy,t,w1,keep,tracks,offsets,config,settings,spec["radius"])
    all_tracks = augment_tracks(tracks,data,xy,multi,keep,prod)
    added_funnel, multiband = source_funnel(data,xy,t,w1,multi,all_tracks,offsets,config,settings,
                                           spec["radius"], reference_keep=keep)
    deltas = defaultdict(list)
    for key in set(r_result) & set(multiband):
        _,a,_,tag,prior,_ = r_result[key]
        _,b,_,_,post,_ = multiband[key]
        deltas[tag].append((len(set(night[post[nsc[post]]]))-len(set(night[prior[nsc[prior]]])),
                           np.ptp(t[post[nsc[post]]])-np.ptp(t[prior[nsc[prior]]]),np.linalg.norm(a-b)))
    added = Counter(multiband[k][3] for k in set(multiband)-set(r_result))
    lost = Counter(r_result[k][3] for k in set(r_result)-set(multiband))
    multiband_summary = dict(W0_r=r_base,W1_r=comparison_funnel,W1_multiband=added_funnel,
                             qualified_band_groups=sorted(verified_band), extra_detection_count=int((multi&~keep).sum()),
                             common={k:dict(n=len(v),extra_nsc_nights_span_years_pm_shift=qtile(v)) for k,v in deltas.items()},
                             new_final=added, lost_final=lost)
    print("Blind r and multiband comparisons complete.",flush=True)
    coverage = dict(scope="targeted_stripe_input" if stripe else "v05_regional_r_input",
                    source=source_coverage(data,xy,tracks,nsc,keep,night,support,spec,settings),
                    area=geometry)
    arrays = keys,groups,corrected,support,nsc,w1,common
    controls = fit_controls(work,center,spec,data,xy,t,sigma,keep,multi,offsets,arrays,
                            tracks,base,config,custom)
    return dict(errors=dict(families=errors,calibration=calinfo,
                            nsc_exposure_metadata_matched=int(np.count_nonzero(nsc & instruments.ne("unknown"))),
                            nsc_detections=int(nsc.sum())),
                controls=controls, coverage=coverage, multiband=multiband_summary,
                resources=timing_resources(work,wall,cpu))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work",type=Path)
    parser.add_argument("region")
    parser.add_argument("inputs",type=Path)
    parser.add_argument("footprint_index",type=Path)
    parser.add_argument("--config",type=Path,default=Path("configs/astrometry-v06.toml"))
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("Use a Slurm compute node.")
    custom = tomllib.loads(args.config.read_text())
    conf = args.config.with_name(custom["frozen_production"])
    config,settings = configuration(conf)
    settings = dict(settings, **{k:custom[k] for k in ("seed","bootstrap_draws","practical_gain","zero_point_limit_masyr")})
    custom = dict(custom, **{k:v for k,v in settings.items() if k not in custom})
    stripe = args.region=="s82_east"
    if args.region not in custom["fields"]:
        parser.error("Use a frozen field.")
    spec = (configuration(conf.with_name("precision-v04.toml"))[1] if stripe else settings)["fields"][args.region]
    work = args.work
    work.mkdir(parents=True,exist_ok=True)
    names = ["detections.csv","external_labels.csv","spectroscopic_qso.csv","validation/gaia.csv"]
    links = {name:args.inputs/name for name in names}
    links.update({"footprint.fits":args.footprint_index,"production_config.toml":conf,
                  "m3_config.toml":conf.with_name(settings["frozen_model"])})
    if stripe:
        links["stripe_config.toml"] = conf.with_name("precision-v04.toml")
    for name, path in links.items():
        target = path.resolve(strict=True)
        link = work/name
        link.parent.mkdir(parents=True,exist_ok=True)
        if not link.exists():
            link.symlink_to(target)
        if link.resolve()!=target:
            parser.error("Use a separate v0.6 directory for different inputs.")
    with run_stage(work,args.config,args.region,"exposure-metadata-v06", inputs=["detections.csv"],
                   outputs=["exposures.csv"]) as go:
        if go:
            data = pd.read_csv(work/"detections.csv",usecols=["survey","exposure_id"])
            exposure_metadata(work,data)
    with run_stage(work,args.config,args.region,"astrometry-v06", inputs=[*links,"exposures.csv"],
                   outputs=["astrometry_summary.json"]) as go:
        if go:
            results = run(work,args.region,config,settings,custom,spec,work/"footprint.fits",stripe)
            save_json(work/"astrometry_summary.json",results)


if __name__=="__main__":
    main()
