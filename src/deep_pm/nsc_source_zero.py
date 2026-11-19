"""Assess blind catalogue QSO zero points at fixed sampling and membership."""
import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from .__main__ import tomllib
from .allband_qso import evaluated_fit
from .astrometry import project, years
from .continuous_reference import SPATIAL, bind, read_offsets, reference_frame, verify_directory
from .exposure_catalogues import inside
from .io import digest, save_json
from .nsc_comparison import comparison
from .nsc_science_summary import SETTINGS
from .pilot_production import valid_epochs
from .precision_v03 import fit_with_calibration
from .provenance import run_stage
from .public_summary import check_aggregate


def fit_r_subset(data, row, keys, solution, center, prod):
    selection = data.band.eq('r').to_numpy()
    nsc = data.survey.eq('NSC').to_numpy()
    time = years(data.observation_time)
    xy = project(data.ra_deg,data.dec_deg,center)
    native = data[['sigma_east_mas','sigma_north_mas']].to_numpy()
    sigma = np.hypot(native,50)
    index = np.flatnonzero(selection)
    if not len(index) or not np.any(nsc[index]):
        return ['no_nsc_r_member'], None
    if nsc[index].all() or len(index)<2 or np.ptp(time[index])<=0:
        return ['underidentified_r_fit'], None
    fit = evaluated_fit(index,data,xy,time,sigma,keys,solution,center,row['reference_epoch'])
    reasons = []
    if not valid_epochs(index,nsc,time,prod):
        reasons.append('time_support')
    model, formal = fit_with_calibration(xy[index],time[index],sigma[index],row['reference_epoch'],[solution[k] for k in keys[index]])
    if formal.max()>prod['max_axis_pm_error_masyr']:
        reasons.append('formal_error')
    if np.quantile(np.linalg.norm(model.residual/sigma[index],axis=1),.68)>3:
        reasons.append('residual')
    return reasons or ['all_subset_fit_gates_pass'],(fit['pm_icrs_masyr'],fit['pm_error_masyr'])


def matched_pairs(extra, existing, use_color=True):
    # Same core and exact run-count category. Maximize feasible pair count,
    # then minimize normalized covariate distance, with no motion in the cost.
    if not extra or not existing:
        return []
    limits = np.array([.20,.50,.70,2.,.20,.20])
    indices = np.arange(len(limits)) if use_color else np.array([0,2,3,4,5])
    a = np.asarray([x['features'] for x in extra])[:,indices]
    b = np.asarray([x['features'] for x in existing])[:,indices]
    delta = abs(a[:,None]-b)/limits[indices]
    separation = np.linalg.norm(np.asarray([x['sky_xy'] for x in extra])[:,None]-np.asarray([x['sky_xy'] for x in existing]),axis=2)/1800000
    valid = np.isfinite(delta).all(axis=2) & (delta<=1).all(axis=2) & (separation<=1)
    cost = np.where(valid,(delta**2).sum(axis=2)+separation**2,1e6)
    x,y = linear_sum_assignment(cost)
    return [(extra[i],existing[j]) for i,j in zip(x,y) if valid[i,j]]


def summarize(samples):
    n = len(samples)
    out = dict(n=n)
    if n<5:
        return out
    for name in ('sdss_runs','sdss_span','nsc_exposures','nsc_nights','nsc_seasons','nsc_span','joint_span','maximum_group_leverage','maximum_night_leverage','shape','neighbor_arcsec','sdss_color','r_formal','reference_fraction','nsc_blend_fraction'):
        data = np.asarray([r[name] for r in samples],float)
        good = np.isfinite(data)
        out[name] = dict(available=int(good.sum()),p16_p50_p84=np.quantile(data[good],[.16,.50,.84]) if good.sum()>=5 else None)
    out['instrument_combinations'] = dict(Counter('+'.join(r['instruments']) for r in samples))
    out['filter_combinations'] = dict(Counter('+'.join(r['bands']) for r in samples))
    out['fit_metrics'] = comparison([{'all':r['all']} for r in samples],SETTINGS)
    return out


def run(work, state, audit, geometry):
    paths = json.loads((state/'fits.json').read_text())
    audit_rows = json.loads((audit/'private_blind_audit.json').read_text())
    grouped = defaultdict(list)
    samples = []
    diagnostics = defaultdict(Counter)
    prod = tomllib.loads(SPATIAL.read_text())['production']
    specs = tomllib.loads(geometry.read_text())['fields']
    for q in audit_rows:
        if q['magnitude'] is not None and 21<=q['magnitude']<23:
            grouped[q['origin_core']].append(q)
    for core,rows in grouped.items():
        folder = Path(paths[core])
        source = (folder/'source_receipt.json').resolve().parent
        receipt = verify_directory(source,('detections.csv','spectra_union.csv'))
        graph = (folder/'reference_receipt.json').resolve().parent/'private_c0.npz'
        solution = read_offsets(graph)
        center = reference_frame(graph)
        frozen_ids = {d for q in rows for d in q['record']['detection_ids']}
        member_parts = []
        context = []
        # One nearest-neighbour query per QSO using positions on its actual SDSS
        # physical run. Crowding is measured after excluding the target's members.
        qso = pd.read_csv(source/'spectra_union.csv')
        qso = qso.loc[inside(qso.ra,qso.dec,specs[core])].reset_index(drop=True)
        columns = ['detection_id','survey','band','ra_deg','dec_deg','observation_time','color','morphology','quality_flags',
                   'sigma_east_mas','sigma_north_mas','exposure_id','nchild']
        for chunk in pd.read_csv(source/'detections.csv',usecols=lambda c:c in columns,chunksize=250000):
            member_parts.append(chunk.loc[chunk.detection_id.isin(frozen_ids)])
            context.append(chunk.loc[chunk.survey.eq('SDSS'),['detection_id','ra_deg','dec_deg','exposure_id']])
        members = pd.concat(member_parts,ignore_index=True).set_index('detection_id',verify_integrity=True)
        context = pd.concat(context,ignore_index=True).dropna(subset=['ra_deg','dec_deg'])
        trees = {}
        for exp,frame in context.groupby('exposure_id'):
            trees[exp] = cKDTree(project(frame.ra_deg,frame.dec_deg,center)),frame.detection_id.to_numpy()
        for q in rows:
            r = q['record']
            local = members.loc[r['detection_ids']]
            nsc = local.survey.eq('NSC')
            info = r['information']
            if len(local)!=r['used_count']:
                raise ValueError('Missing frozen detections')
            late,early = local.loc[nsc],local.loc[~nsc]
            shape_values = late.morphology.dropna()
            shape = shape_values.loc[shape_values.between(0,1)].median()
            color = early['color'].median() if 'color' in early else float('nan')
            qpos = qso.iloc[q['qso_local_index']]
            sky = project([qpos.ra],[qpos.dec],center)[0]
            near = []
            for exp in early.exposure_id.unique():
                if exp not in trees:
                    continue
                tree,did = trees[exp]
                dist, ids_local = tree.query(sky,k=min(5,len(did)))
                near.extend(d/1000 for d,j in zip(np.atleast_1d(dist),np.atleast_1d(ids_local)) if did[j] not in local.index)
            blend = np.mean((pd.to_numeric(late.quality_flags,errors='coerce').fillna(0).astype(int)&3)!=0)
            k = r['kind'],int(q['magnitude'])
            rfit = None
            if r['kind']=='extra_r_anchor':
                causes,rfit = fit_r_subset(local,r,r['physical_groups'],solution,center,prod)
                diagnostics[k].update(causes)
            d = dict(core=core,kind=r['kind'],mag=q['magnitude'],all=q['values']['blind_all'],rfit=rfit,
                     sdss_runs=early.exposure_id.nunique(),sdss_span=np.ptp(early.observation_time)/365.25,
                     nsc_exposures=info['nsc_exposures'],nsc_nights=info['nsc_nights'],nsc_seasons=info['nsc_seasons'],
                     nsc_span=info['nsc_span_years'],joint_span=info['joint_span_years'],
                     maximum_group_leverage=max(info['largest_information_fraction']['physical_group']),
                     maximum_night_leverage=max(info['largest_information_fraction']['observing_night']),
                     shape=shape,neighbor_arcsec=min(near,default=float('nan')),sdss_color=color,nsc_blend_fraction=blend,
                     r_formal=max(r['pm_error_masyr']),instruments=r['instruments'],bands=r['bands'],
                     reference_fraction=np.trace(np.asarray(r['reference_tangent_covariance'])[1::2,1::2])/
                     np.trace((np.asarray(r['reference_tangent_covariance'])+np.asarray(r['independent_tangent_covariance']))[1::2,1::2]),
                     sky_xy=sky)
            d['features'] = [q['magnitude'],color,np.log1p(d['nsc_span']),d['joint_span'],d['maximum_group_leverage'],shape]
            samples.append(d)
        print(core,len(rows),digest(receipt),flush=True)
    result = {}
    for low in (21,22):
        sample = [r for r in samples if low<=r['mag']<low+1]
        entry = {}
        for kind in ('r_existing','extra_r_anchor','non_r_anchor'):
            part = [r for r in sample if r['kind']==kind]
            entry[kind] = summarize(part)
            entry[kind]['space'] = {core:comparison([{'all':r['all']} for r in part if r['core']==core],SETTINGS)
                for core in sorted({r['core'] for r in part})}
        rpart = [r for r in sample if r['kind']=='extra_r_anchor' and r['rfit'] is not None]
        entry['physical_r_subsets_of_extra'] = dict(gate_counts=dict(diagnostics['extra_r_anchor',low]),
            fit=comparison([{'all':r['all'],'r_subset':r['rfit']} for r in rpart],SETTINGS,(('all','r_subset'),)))
        for use_color in (True,False):
            matched = []
            blocks = sorted({(r['core'],r['sdss_runs']==1) for r in sample})
            for block in blocks:
                cohort = [r for r in sample if (r['core'],r['sdss_runs']==1)==block]
                matched.extend(matched_pairs([r for r in cohort if r['kind']=='extra_r_anchor'],
                                             [r for r in cohort if r['kind']=='r_existing'],use_color))
            pairs = [{'extra':a['all'],'existing':b['all']} for a,b in matched]
            entry['matched_with_color' if use_color else 'matched_color_unrestricted'] = dict(
                comparison=comparison(pairs,SETTINGS,(('extra','existing'),)),
                extra=summarize([a for a,b in matched]),existing=summarize([b for a,b in matched]),
                contributing_cores=len({a['core'] for a,b in matched}))
        result['r'+str(low)] = entry
    out = dict(bins=result,slurm_job=os.environ['SLURM_JOB_ID'],scope='J2015.5 blind IDs frozen before spectroscopy. One-to-one without replacement within core and single/multiple-run category. Calipers: 0.5 deg, 0.20 mag r, 0.50 g-i, 0.70 log(1+NSC span), 2 yr joint span, 0.20 leverage, 0.20 class_star. Match uses no PM or error outcomes. Neighbor distance proxies crowding in retrieved SDSS detections, not an image blend truth. r subsets reuse exactly the selected source r detections; fit-only gate exits do not reproduce upstream r association. Differences remain conditional and can reflect unobserved selection.')
    check_aggregate(out)
    save_json(work/'source_zero_summary.json',out)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('work','state','audit','geometry'):
        p.add_argument(k,type=Path)
    a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        p.error('Use Slurm')
    a.work.mkdir(parents=True,exist_ok=True)
    bind(a.work,'frozen_receipt.json',verify_directory(a.state,('fits.json','private_owner_overlay.json')))
    bind(a.work,'audit_receipt.json',verify_directory(a.audit,('private_blind_audit.json',)))
    with run_stage(a.work,a.geometry,'C','nsc-source-zero-v16',('frozen_receipt.json','audit_receipt.json'),('source_zero_summary.json',)) as go:
        if go:
            run(a.work,a.state,a.audit,a.geometry)


if __name__=='__main__':
    main()
