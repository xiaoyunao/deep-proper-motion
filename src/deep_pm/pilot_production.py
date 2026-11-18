"""Blind fixed-position trial catalogue and measured source-selection funnel."""
import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import resource
from time import monotonic, process_time

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from . import access
from .__main__ import detection_arrays, tomllib
from .associate import motion_tracks, static_tracks
from .astrometry import fit_motion, project, unproject
from .dark_precision import bin_name, query_unique
from .io import save_json
from .nsc_comparison import configuration
from .precision_v03 import calibration, calibration_groups, calibration_support, fit_with_calibration
from .provenance import run_stage


STAGES = ("initial", "sdss_nearby", "associated", "fit_eligible", "calibrated", "final")


def quality_detections(work, center, config, production):
    options = dict(config)
    options.pop("smoke_max_mag")
    options["smoke_max_error_mas"] = config["calibration"]["astrometric_max_error_mas"]
    data, xy, t, sigma = detection_arrays(work, center, options)
    keep = (data.band.eq("r") & data.magnitude.notna()
            & data.magnitude_error.between(0, 2.5/(np.log(10)*production["min_snr"]), inclusive="right"))
    return data, xy, t, sigma, keep.to_numpy()


def make_tracks(xy, exp, keep, radius):
    selected = np.flatnonzero(keep)
    return [selected[list(item)] for item in static_tracks(xy[selected], exp[selected], radius)]


def valid_epochs(index, nsc, time, settings):
    late = index[nsc[index]]
    return (len(late)>=2 and len(late)<len(index)
            and np.ptp(time[late])>=settings["min_nsc_span_years"]
            and np.ptp(time[index])>=settings["min_joint_span_years"])


ANCHOR_EXITS = ("outside_core", "pointlike_morphology", "no_sdss_position", "anchor_competition",
                "time_baseline", "m3_support", "quality", "final")


def update_anchor_trace(trace, slots, status):
    if trace is None:
        return
    for slot in slots:
        if slot >= 0 and (slot not in trace or ANCHOR_EXITS.index(trace[slot]) < ANCHOR_EXITS.index(status)):
            trace[slot] = status


def source_funnel(data, xy, time, sigma, keep, tracks, offsets, config, settings, radius, reference_keep=None, trace=None, fits=None, initial=None, anchor_info=None, model_cache=None, observation_groups=None, core_members=None, quality_audit=None):
    prod = settings["production"]
    group = calibration_groups(data) if observation_groups is None else observation_groups
    nsc = data.survey.eq("NSC").to_numpy()
    exposure = data.exposure_id.to_numpy()
    owner = np.full(len(xy), -1, dtype=int)
    for j, members in enumerate(tracks):
        owner[members] = j
    sdss_tree = cKDTree(xy[keep & ~nsc])
    reference_keep = keep if reference_keep is None else reference_keep
    if initial is None:
        initial = make_tracks(xy, exposure, reference_keep & nsc, prod["match_radius_mas"])
    support = calibration_support(group,offsets)
    morphology, mag = data.morphology.to_numpy(), data.magnitude.to_numpy()
    p = {}
    limit = np.tan(np.deg2rad(radius))*180/np.pi*3_600_000
    results = {}
    for anchor in initial:
        position = np.median(xy[anchor],axis=0)
        if trace is not None:
            touched = np.unique(owner[anchor])
        if (np.linalg.norm(position)>limit if core_members is None else not core_members(position)):
            if trace is not None:
                update_anchor_trace(trace,touched,"outside_core")
            continue
        shape = np.median(morphology[anchor])
        if shape<prod.get('min_anchor_morphology', .8):
            if trace is not None:
                update_anchor_trace(trace,touched,"pointlike_morphology")
            continue
        bucket = bin_name(np.median(mag[anchor]),settings["reference_r_edges"])
        stats = p.setdefault(bucket, Counter({step:0 for step in STAGES}))
        stats["initial"] += 1
        if sdss_tree.query(position)[0] > prod["match_radius_mas"]:
            stats["no_sdss_within_fixed_gate"] += 1
            if trace is not None:
                update_anchor_trace(trace,touched,"no_sdss_position")
            continue
        stats["sdss_nearby"] += 1
        if trace is not None:
            update_anchor_trace(trace,touched,"anchor_competition")
        overlap = Counter(owner[anchor])
        j, covered = overlap.most_common(1)[0]
        if j<0:
            stats["split_or_competing_track"] += 1
            continue
        index = tracks[j]
        if (covered/len(anchor)<prod["validation_min_recovery"]
                or covered<=np.count_nonzero(nsc[index] & reference_keep[index])/2 or nsc[index].all()):
            stats["split_or_competing_track"] += 1
            continue
        if j in results:
            raise ValueError("Two independent NSC anchors own one final track.")
        stats["associated"] += 1
        update_anchor_trace(trace,[j],"time_baseline")
        if not valid_epochs(index,nsc,time,prod):
            stats["insufficient_time_sampling"] += 1
            continue
        stats["fit_eligible"] += 1
        update_anchor_trace(trace,[j],"m3_support")
        use = index[support[index]]
        if not valid_epochs(use,nsc,time,prod):
            stats["insufficient_calibration_coverage"] += 1
            continue
        stats["calibrated"] += 1
        update_anchor_trace(trace,[j],"quality")
        cached = model_cache.get(j) if model_cache is not None else None
        if cached is not None:
            _, model, error = cached
            if not np.array_equal(use, cached[0]):
                raise ValueError("Trial must use the same calibrated detections for a cached fit")
        else:
            model, error = fit_with_calibration(xy[use],time[use],sigma[use],config["reference_epoch"],
                                                [offsets[k] for k in group[use]])
            if model_cache is not None:
                model_cache[j] = (use, model, error)
        residual = np.quantile(np.linalg.norm(model.residual/sigma[use],axis=1),.68)
        finite = np.isfinite(error).all()
        formal = not finite or error.max()>prod["max_axis_pm_error_masyr"]
        scatter = residual>3
        if quality_audit is not None and bucket=="22_to_23":
            weights = model.weight / np.mean(sigma[use]**2,axis=1)
            lever = weights*(time[use]-np.average(time[use],weights=weights))**2
            first = use[np.argmin(time[use])]
            first_run_share = lever[exposure[use]==exposure[first]].sum()/lever.sum()
            late = use[nsc[use]]
            quality_audit.append((formal,scatter,finite,error.max(),residual,np.ptp(time[use]),
                                  first_run_share,len(set(exposure[late])),np.median(mag[late])))
        if formal or scatter:
            stats["final_quality_rejection"] += 1
            continue
        stats["final"] += 1
        update_anchor_trace(trace,[j],"final")
        if fits is not None:
            fits[j] = model
        if anchor_info is not None:
            anchor_info[j] = (shape, position, anchor)
        results[j] = (model.coeff[0], model.coeff[1], error, bucket, use, len(index))
    return p, results


def held_year(data, xy, t, sigma, keep, offsets, config, settings):
    from astropy.time import Time

    prod = settings["production"]
    year = prod["holdout_year"]
    boundaries = Time([f"{year}-01-01",f"{year+1}-01-01"],scale="utc").tt.jyear
    nsc = data.survey.eq("NSC").to_numpy()
    held = nsc & (t>=boundaries[0]) & (t<boundaries[1]) & keep
    visits_seen = data.exposure_id.loc[held].nunique()
    if not visits_seen:
        return dict(year=year, held_visits=0, by_r={})
    groups = calibration_groups(data)
    keep_cal = calibration_support(groups,offsets)
    train = make_tracks(xy,data.exposure_id.to_numpy(),keep & ~held & keep_cal,prod["match_radius_mas"])
    epoch = config["reference_epoch"]
    estimates, corrected, bins = [], [], []
    mag = data.magnitude.to_numpy()
    shape = data.morphology.to_numpy()
    for index in train:
        late = index[nsc[index]]
        if not valid_epochs(index,nsc,t,prod) or np.median(shape[late])<.8 or np.median(mag[late])<20:
            continue
        estimates.append(fit_motion(xy[index],t[index],sigma[index],epoch).coeff)
        changes = np.array([offsets[g][0] for g in groups[index]])
        corrected.append(fit_motion(xy[index]-changes,t[index],sigma[index],epoch).coeff)
        bins.append(bin_name(np.median(mag[late]),settings["reference_r_edges"]))
    if not bins:
        return dict(year=year, held_visits=int(visits_seen), by_r={})
    res = defaultdict(list)
    supported_visits = 0
    a, b = np.asarray(estimates), np.asarray(corrected)
    for key, subset in data.loc[held & keep_cal].groupby("exposure_id"):
        ix = subset.index.to_numpy()
        if not len(ix):
            continue
        supported_visits += 1
        midpoint = np.median(t[ix])
        pred = a[:,0]+a[:,1]*(midpoint-epoch)
        hit, closest = query_unique(cKDTree(xy[ix]),pred,500,prod["competitor_margin_mas"])
        for j in np.flatnonzero(hit):
            i = ix[closest[j]]
            before = xy[i] - a[j,0] - a[j,1]*(t[i]-epoch)
            after = xy[i] - offsets[groups[i]][0] - b[j,0] - b[j,1]*(t[i]-epoch)
            res[bins[j]].append((j,np.linalg.norm(before),np.linalg.norm(after)))
    by_r = {}
    for tag in sorted(set(bins)):
        part = res[tag]
        sample = len({r[0] for r in part})
        by_r[tag] = dict(predicted_sources=bins.count(tag), matched_sources=sample,
                         matched_visit_positions=len(part),
                         paired_radial_p68_mas=np.quantile(np.array(part)[:,1:],.68,axis=0)
                         if sample >= settings["min_publish_count"] else None)
    return dict(year=year, held_visits=int(visits_seen), supported_visits=supported_visits, by_r=by_r)


def old_motion_window(data, xy, t, sigma, keep, settings, frozen_path):
    config = tomllib.loads(frozen_path.read_text())
    radius = settings["production"]["comparison_radius_deg"]*3_600_000
    use = np.flatnonzero((np.linalg.norm(xy,axis=1)<radius) & keep)
    exposure = data.exposure_id.to_numpy()[use]
    a = monotonic()
    static = static_tracks(xy[use],exposure,config["fixed_radius_mas"])
    nsc = data.survey.eq("NSC").to_numpy()[use]
    def two(rows):
        subset = nsc[list(rows)]
        return int(subset.any() and not subset.all())
    comparable = dict(detections=len(use), area_sqdeg=float(np.pi*(radius/3_600_000)**2),
                      static_cross_survey=sum(two(g) for g in static))
    if len(use)<2:
        return comparable
    try:
        selected, _, _, summary = motion_tracks(xy[use],t[use],sigma[use],exposure,static,config)
    except (RuntimeError,ValueError) as exc:
        return dict(comparable,status=type(exc).__name__, seconds=monotonic()-a)
    return dict(comparable, status="computed", motion_cross_survey=sum(two(g.indices) for g in selected),
                motion_secure_cross_survey=sum(two(g.indices) for g in selected if g.status=="secure"),
                motion_difficult_components=summary.get("difficult_components"), seconds=monotonic()-a)


def timing_resources(work, wall, cpu):
    return dict(cpu_seconds=process_time()-cpu, wall_seconds=monotonic()-wall,
                peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                detection_bytes=(work/"detections.csv").stat().st_size)


def main():
    from .pilot_validation import validate_catalogue

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("work",type=Path)
    p.add_argument("region")
    p.add_argument("inputs",type=Path)
    p.add_argument("--config",type=Path,default=Path("configs/production-v05.toml"))
    args = p.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        p.error("Use a Slurm compute node.")
    config, settings = configuration(args.config)
    spec = settings["fields"][args.region]
    center = spec["ra"],spec["dec"]
    work = args.work
    work.mkdir(parents=True,exist_ok=True)
    for name in ("detections.csv","external_labels.csv","spectroscopic_qso.csv","validation","sdss.csv"):
        link = work/name
        target = (args.inputs/name).resolve(strict=True)
        if not link.exists():
            link.symlink_to(target)
        if link.resolve()!=target:
            p.error("Use a separate production directory for changed archive inputs.")
    wall, cpu = monotonic(), process_time()
    prod = settings["production"]
    data,xy,t,sigma,keep = quality_detections(work,center,config,prod)
    with run_stage(work,args.config,args.region,"blind-associate-v05",inputs=["detections.csv"],
                   outputs=["private_tracks.json"]) as go:
        if go:
            start = monotonic()
            tracks = make_tracks(xy,data.exposure_id.to_numpy(),keep,prod["match_radius_mas"])
            save_json(work/"private_tracks.json",dict(tracks=tracks,association_seconds=monotonic()-start))
    saved = json.loads((work/"private_tracks.json").read_text())
    tracks = [np.asarray(x,dtype=int) for x in saved["tracks"]]
    needs = ["detections.csv","external_labels.csv","validation/gaia.csv","spectroscopic_qso.csv","private_tracks.json"]
    optional = "validation/external_pm.csv"
    if (work/optional).is_file():
        needs.append(optional)
    outputs = ["production_summary.json","private_trial_catalogue.csv","private_qso_pairs.json"]
    with run_stage(work,args.config,args.region,"production-v05",inputs=needs,outputs=outputs) as go:
        if not go:
            return
        print("Blind association frozen. Solving independent galaxy offsets.",flush=True)
        qso = pd.read_csv(work/"spectroscopic_qso.csv")
        offsets, cal_info = calibration(work,center,config,project(qso.ra,qso.dec,center))
        cal_info.pop("sdss_run_camcol",None)
        funnel, selected = source_funnel(data,xy,t,sigma,keep,tracks,offsets,config,settings,spec["radius"])
        print(f"{len(selected):,} trial sources; starting independent validation",flush=True)
        validation = validate_catalogue(work,center,spec["radius"],data,xy,t,sigma,keep,tracks,selected,
                                        offsets,qso,config,settings)
        table = []
        for identifier,(pos,pm,err,tag,used,n) in selected.items():
            ra,dec = unproject(np.array([pos]),center)[0]
            table.append((identifier,ra,dec,*pm,*err,tag,len(used),n, np.ptp(t[used])))
        columns = ["track_id","ra","dec","pm_x_tangent","pm_y_tangent","sigma_east","sigma_north",
                   "r_bin","fit_detections","associated_detections","joint_baseline"]
        pd.DataFrame(table,columns=columns).to_csv(work/"private_trial_catalogue.csv",index=False)
        held = held_year(data,xy,t,sigma,keep,offsets,config,settings)
        legacy = old_motion_window(data,xy,t,sigma,keep,settings,args.config.with_name("experiment-a-v02.toml"))
        area = 2*np.pi*(1-np.cos(np.deg2rad(spec["radius"])))*(180/np.pi)**2
        stats = timing_resources(work,wall,cpu)
        stats.update(association_seconds=saved["association_seconds"], input_detections=len(data),
                     usable_r_detections=int(keep.sum()))
        save_json(work/"production_summary.json",dict(core_area_sqdeg=area, funnel=funnel,
                  final_per_sqdeg=sum(v["final"] for v in funnel.values())/area, calibration=cal_info,
                  validation=validation, holdout=held, motion_window=legacy, resources=stats))


if __name__=="__main__":
    main()
