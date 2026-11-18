"""Apply a single regional C0 solution to independent spatial W0 cores."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import resource
from time import monotonic

import healpy as hp
import numpy as np
import pandas as pd
from astropy.time import Time

from .__main__ import tomllib
from .access import Cache
from .astrometry import fit_motion, project, unproject
from .continuous_reference import FILES, PRECISION, SPATIAL, bind, read_offsets, reference_frame, verify_directory
from .exposure_catalogues import inside
from .exposure_pilot import contrast_samples, matched_legacy
from .io import digest, save_json
from .nsc_comparison import qso_matches
from .pilot_production import make_tracks, quality_detections, source_funnel, valid_epochs
from .precision_v03 import calibration_groups,calibration_support
from .provenance import run_stage
from .regional_production import prediction, record_catalogue
from .regional_validation import official_table
from .standard_motion import calibrated_covariance, icrs_motion
from .validation import gaia_labels, groups_by_label
from .validation_audit import anchored_bright_gaia, gaia_bright, gaia_observation_quality, quantiles


def groups_for_removal(visits, times, nsc):
    night = np.floor((times-2000)*365.25).astype(np.int32)
    return {
        "sdss_run": np.where(nsc,"",visits),
        "nsc_exposure": np.where(nsc,visits,""),
        "nsc_night": np.where(nsc,night.astype(str),""),
    }


def group_deletions(truth, data, xy, t, sigma, keep, nsc, keys, solution, settings, config, magnitude):
    support = calibration_support(keys,solution)
    positions = xy - np.array([solution[key][0] if key in solution else np.zeros(2) for key in keys])
    visits = data.exposure_id.to_numpy()
    group = groups_for_removal(visits,t,nsc)
    prod = settings["production"]
    changes = []
    for k, detections in truth.items():
        if not 21 <= magnitude[k] < 23:
            continue
        ix = detections[support[detections]&keep[detections]]
        if not valid_epochs(ix,nsc,t,prod):
            continue
        baseline = fit_motion(positions[ix],t[ix],sigma[ix],config["reference_epoch"]).coeff[1]
        leave = {}
        for family, labels in group.items():
            leave[family] = {}
            for g in sorted(set(labels[ix])- {""}):
                left = ix[labels[ix]!=g]
                leave[family][g] = (fit_motion(positions[left],t[left],sigma[left],config["reference_epoch"]).coeff[1]
                                    if valid_epochs(left,nsc,t,prod) else None)
        changes.append(dict(magnitude=magnitude[k], baseline=baseline, removed=leave))
    return changes


def night_partition(data,parity):
    """Metadata-only whole-NSC-night split; every SDSS observation is shared."""
    nsc=data.survey.eq('NSC').to_numpy()
    utc=Time(data.observation_time.to_numpy(),format='mjd',scale='tt').utc.mjd
    night=np.floor(utc-.75).astype(int)
    selected=np.unique(night[nsc])[parity::2]
    return ~nsc|np.isin(night,selected)


def fit_core(work, source, spec, shared, legacy, fit_annual, diagnostic_covariance=False,nsc_night_half=None):
    start = monotonic()
    center = reference_frame(shared)
    config = tomllib.loads(PRECISION.read_text())
    settings = tomllib.loads(SPATIAL.read_text())
    settings.update(bootstrap_draws=400,bootstrap_absolute_zero=True)
    prod = settings["production"]
    solution, gauges = read_offsets(shared,with_gauge=True)
    data,xy,t,sigma,keep = quality_detections(source,center,config,prod)
    if nsc_night_half is not None:
        keep &= night_partition(data,nsc_night_half)
    keys = calibration_groups(data)
    nsc = data.survey.eq("NSC").to_numpy()
    visits = data.exposure_id.to_numpy()
    visit_times=data.loc[keep&nsc,['exposure_id','observation_time']].drop_duplicates('exposure_id')
    visit_nights=dict(zip(visit_times.exposure_id,np.floor(Time(visit_times.observation_time.to_numpy(),format='mjd',scale='tt').utc.mjd-.75).astype(int)))
    on_core = inside(data.ra_deg,data.dec_deg,spec)
    available = pd.Series(keys[on_core&keep]).map(gauges)
    counts = available.value_counts()
    main_gauge = min(counts.index,key=lambda k:(-counts[k],k)) if len(counts) else None
    solution = {k:v for k,v in solution.items() if gauges[k]==main_gauge}
    def core(position):
        return inside(*unproject(position,center)[0],spec)

    tracks = make_tracks(xy,visits,keep,prod["match_radius_mas"])
    seeds = make_tracks(xy,visits,nsc&keep,prod["match_radius_mas"])
    models = {}
    funnel,products = source_funnel(data,xy,t,sigma,keep,tracks,solution,config,settings,9.,
                          initial=seeds,observation_groups=keys,core_members=core,fits=models)
    # The final catalogue uses half-open ownership at the common reference epoch.
    owned = {k:v for k,v in products.items() if core(v[0])}
    nights = np.floor((t-2000)*365.25).astype(int)
    diagnostics = record_catalogue(work,center,data,xy,t,sigma,keys,visits,nights,nsc,solution,
                                   {"W0":(tracks,owned,None)},models,config,settings)
    standard = []
    for slot, (pos,pm,err,tag,used,_) in owned.items():
        covariance, independent, common, influence = calibrated_covariance(models[slot],t[used],sigma[used],
                                              [solution[k] for k in keys[used]])
        if not np.allclose(np.sqrt(np.diag(covariance)[[1,3]]),err,rtol=1e-9,atol=1e-10):
            raise ValueError('Full covariance must preserve frozen tangent PM errors')
        entry = dict(track_id=slot,r_bin=tag,projection_center_deg=center, **icrs_motion(pos,pm,center,covariance),
            exposures=sorted(set(visits[used])),nsc_nights_utc18=sorted({visit_nights[e] for e in visits[used] if e.startswith('NSC:')}),
            tangent_covariance=covariance,independent_tangent_covariance=independent,
            reference_tangent_covariance=common, detection_ids=data.detection_id.iloc[used].tolist())
        if diagnostic_covariance:
            samples = np.stack([solution[k][1] for k in keys[used]],axis=1)
            entry.update(tangent_state=models[slot].coeff.T.ravel(),response=influence,
                reference_state_draws=-np.einsum('pna,dna->dp',influence,samples),
                measurement_sigma_mas=sigma[used],physical_groups=keys[used])
        standard.append(entry)
    import gzip
    from .io import json_default
    with gzip.open(work/'private_standard_motion.json.gz','wt') as stream:
        json.dump(standard,stream,default=json_default,allow_nan=False)
    original = pd.read_csv(source/"external_labels.csv",dtype=str).set_index("detection_id").original_object_id
    official = original,official_table(source,{})
    qso = pd.read_csv(source/"spectra_union.csv")
    qso = qso.loc[inside(qso.ra,qso.dec,spec)].reset_index(drop=True)
    projected = project(qso.ra,qso.dec,center)
    labels,truth = qso_matches(data,xy,keep,projected)
    archive = pd.read_csv(legacy)
    archive = archive.loc[inside(archive.RA,archive.DEC,spec,.06)]
    archive_qso,_ = matched_legacy(archive,projected,center)
    ref = np.zeros((len(qso),2))
    mag = qso.reference_r.to_numpy()
    blocks = []
    qso_stats = contrast_samples(truth,labels,data,xy,t,sigma,keep,nsc,settings,config,(keys,),(solution,),
                                  center,mag,qso.origin.to_numpy(),ref,ref,official,archive_qso,
                                  tracks,{"C0":owned},blocks)
    save_json(work/"private_qso_blocks.json",blocks)
    removed = group_deletions(truth,data,xy,t,sigma,keep,nsc,keys,solution,settings,config,mag)
    save_json(work/"private_group_deletions.json",removed)
    gaia = pd.read_csv(source/"validation/gaia.csv")
    if "visibility_periods_used" not in gaia:
        visibility = gaia_observation_quality(Cache(work/"gaia_quality_cache"),spec)
        gaia = gaia.merge(visibility,on="source_id",how="left",validate="one_to_one")
    gaia,bright = anchored_bright_gaia(gaia,seeds,xy,data.magnitude.to_numpy(),center)
    lab,cat = gaia_labels(data.loc[keep].reset_index(drop=True),xy[keep],t[keep],sigma[keep],gaia,center,config)
    label = np.full(len(data),-1)
    label[keep] = lab
    truth = groups_by_label(label,data,t,cat,center,9.,config)
    use = inside(cat.ra,cat.dec,spec)
    magnitude = np.full(len(cat),np.nan)
    for k,ix in truth.items():
        if use[k] and nsc[ix].any():
            magnitude[k] = np.median(data.magnitude.iloc[ix[nsc[ix]]])
    bright_mask = gaia_bright(cat,magnitude)
    truth = {i:idx for i,idx in truth.items() if bright_mask[i]}
    label[~np.isin(label,list(truth))] = -1
    archive_gaia,_ = matched_legacy(archive,project(cat.ra,cat.dec,center),center)
    gaia_stats = contrast_samples(truth,label,data,xy,t,sigma,keep,nsc,settings,config,(keys,),(solution,),
                    center,magnitude,np.repeat("Gaia",len(cat)),cat[["pm_tangent_x","pm_tangent_y"]].to_numpy(),
                    cat[["pmra_error","pmdec_error"]].to_numpy(),official,archive_gaia,tracks,{"C0":owned})
    annual = {}
    if fit_annual:
        multi = keep|(nsc&data.band.isin(["g","i","z"]).to_numpy()&data.magnitude_error.between(0,.21715).to_numpy())
        annual = prediction(work,center,data,xy,t,sigma,sigma,keys,nsc,keep,multi,solution,
                            settings,config,dict(radius=9.,core_members=core))
    on_core = inside(data.ra_deg,data.dec_deg,spec)
    supported = calibration_support(keys,solution)
    used = supported&keep&on_core
    pix = hp.ang2pix(128,data.ra_deg.to_numpy()[used],data.dec_deg.to_numpy()[used],lonlat=True)
    occupied = Counter(map(str,pix))
    positions = unproject([v[0] for v in owned.values()],center) if owned else np.empty((0,2))
    owned_pixel = hp.ang2pix(128,*positions.T,lonlat=True)
    pixel_w0 = Counter(map(str,owned_pixel))
    observations = {}
    for name,valid in (("SDSS",used&~nsc),("NSC",used&nsc)):
        ids = visits[valid]
        observations[name] = dict(n_rows=int(valid.sum()),distinct_visits=len(set(ids)), nights=len(set(nights[valid])),
            span_years=float(np.ptp(t[valid])) if valid.any() else None,unconstrained=int(np.count_nonzero(
                keep&on_core&~supported&(nsc if name=="NSC" else ~nsc))))
    save_json(work/"continuous_core_summary.json",dict(projection_center_deg=center,standard_covariance_count=len(standard),
        w0_funnel=funnel,owned_w0=len(owned),ownership_exits=len(products)-len(owned),
        formal_error_masyr=quantiles([p[2] for p in owned.values()]),observations=observations,gaia=gaia_stats,
        gaia_policy=bright,qso=qso_stats,annual=annual,leverage=diagnostics,w0_ring_counts=pixel_w0,
        supported_ring_measurements=occupied,shared_solution_sha256=digest(shared),
        reference_gauge=main_gauge,other_gauge_r_measurements=int(available.notna().sum()-counts.get(main_gauge,0)),
        usable_error_rows=len(data),
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,elapsed_seconds=monotonic()-start))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("work",type=Path)
    p.add_argument("region")
    for name in ("source","shared","legacy","config"):
        p.add_argument(name,type=Path)
    p.add_argument("--annual",action="store_true")
    p.add_argument("--diagnostic-covariance",action="store_true")
    p.add_argument('--nsc-night-half',type=int,choices=(0,1))
    a = p.parse_args()
    if a.nsc_night_half is not None and a.annual:
        p.error('Night split is a separate validation fit; annual augmentation would restore excluded bands')
    if "SLURM_JOB_ID" not in os.environ:
        p.error("Run on a Slurm compute node")
    a.work.mkdir(parents=True,exist_ok=True)
    options = dict(annual=a.annual,diagnostic_covariance=a.diagnostic_covariance)
    if a.nsc_night_half is not None:options['nsc_night_half']=a.nsc_night_half
    option_path = a.work/'fit_options.json'
    if option_path.exists():
        if json.loads(option_path.read_text())!=options:
            p.error('Fit options changed; use a separate versioned output directory')
    elif (a.work/'run_manifest.json').exists():
        p.error('Historical run has no option snapshot; preserve it and use a new directory')
    else:
        save_json(option_path,options)
    bind(a.work,"source_receipt.json",verify_directory(a.source,FILES))
    bind(a.work,"reference_receipt.json",verify_directory(a.shared,("private_c0.npz",)))
    outputs = ("continuous_core_summary.json","private_qso_blocks.json","private_group_deletions.json","private_catalogue.csv.gz",
               "private_standard_motion.json.gz")
    with run_stage(a.work,a.config,a.region,"continuous-core-v12",inputs=("source_receipt.json","reference_receipt.json","fit_options.json"),
                   outputs=outputs) as go:
        if go:
            spec = tomllib.loads(a.config.read_text())["fields"][a.region]
            fit_core(a.work,a.source,spec,a.shared/"private_c0.npz",a.legacy,a.annual,a.diagnostic_covariance,a.nsc_night_half)


if __name__=="__main__":
    main()
