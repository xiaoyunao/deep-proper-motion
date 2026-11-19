"""Training-year-only all-filter discovery with separate non-r anchors.

Candidate yields are not adopted dark-star counts: shared-error calibration,
identity, blending and absolute zero point remain independent requirements.
"""
import argparse
from collections import Counter,defaultdict
import os
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.time import Time
from scipy.spatial import cKDTree

from . import access
from .__main__ import tomllib
from .allband_qso import physical_candidates,evaluated_fit
from .astrometry import unproject
from .astrometry_audit import augment_tracks
from .condition_holdout import held_prediction_variance
from .continuous_reference import PRECISION,SPATIAL,bind,read_offsets,reference_frame,verify_directory
from .dark_precision import query_unique
from .exposure_catalogues import inside
from .independent_year import choose_year,exclusive
from .io import save_json
from .pilot_production import make_tracks,quality_detections,source_funnel,valid_epochs
from .precision_v03 import calibration_groups,calibration_support,fit_with_calibration
from .provenance import run_stage
from .public_summary import check_aggregate
from .regional_preflight import CHIP_VERTICES
from .regional_production import r_chip_shapes
from .validation import ratio


def r_magnitude(index,data,nsc):
    r=index[nsc[index]&data.band.iloc[index].eq('r').to_numpy()]
    if len(r):return float(np.median(data.magnitude.iloc[r])),'NSC_r'
    early=index[~nsc[index]&data.band.iloc[index].eq('r').to_numpy()]
    return (float(np.median(data.magnitude.iloc[early])),'SDSS_r') if len(early) else (np.nan,'unavailable')


def run(work,source,shared,spec):
    config=tomllib.loads(PRECISION.read_text());settings=tomllib.loads(SPATIAL.read_text());prod=settings['production']
    graph=shared/'private_c0.npz';center=reference_frame(graph);solution,gauges=read_offsets(graph,with_gauge=True)
    data,xy,time,sigma,keep=quality_detections(source,center,config,prod);nsc=data.survey.eq('NSC').to_numpy();keys=calibration_groups(data)
    visit=data.exposure_id.to_numpy();physical=physical_candidates(data)
    votes=pd.Series(keys[keep&inside(data.ra_deg,data.dec_deg,spec)]).map(gauges).value_counts()
    gauge=min(votes.index,key=lambda k:(-votes[k],k)) if len(votes) else None
    solution={k:v for k,v in solution.items() if gauges[k]==gauge};support=calibration_support(keys,solution)
    year,held_year=choose_year(time,visit,nsc,nsc&physical&support)
    if held_year is None:
        save_json(work/'allband_blind_year_summary.json',dict(status='insufficient_calibrated_years',distinct_years=len(set(year[nsc&physical&support])),reference_gauge=gauge));save_json(work/'private_training.json',[]);save_json(work/'private_held.json',[]);return
    held=nsc&(year==held_year);train_r=keep&~held
    eligible=(physical&support)&~held
    core=lambda position:inside(*unproject(position,center)[0],spec)
    tracks=make_tracks(xy,visit,train_r,prod['match_radius_mas']);seeds=make_tracks(xy,visit,train_r&nsc,prod['match_radius_mas'])
    _,baseline=source_funnel(data,xy,time,sigma,train_r,tracks,solution,config,settings,9.,initial=seeds,observation_groups=keys,core_members=core)
    expanded=train_r|(eligible&nsc)
    augmented=augment_tracks(tracks,data,xy,expanded,train_r,prod)
    funnel,products=source_funnel(data,xy,time,sigma,expanded,augmented,solution,config,settings,9.,initial=seeds,reference_keep=train_r,observation_groups=keys,core_members=core)
    candidates=[];used=set();counts=Counter()
    for slot,value in products.items():
        index=value[4];mag,mag_source=r_magnitude(index,data,nsc)
        if not 20<=mag<23:continue
        kind='r_existing' if slot in baseline else 'extra_r_anchor'
        candidates.append(dict(kind=kind,index=index,magnitude=mag,magnitude_source=mag_source,baseline=baseline.get(slot)))
        used.update(index)
    full_tracks=make_tracks(xy,visit,eligible,prod['match_radius_mas'])
    counts['all_filter_training_tracks']=len(full_tracks)
    for index in full_tracks:
        if not nsc[index].any() or nsc[index].all():counts['non_r_missing_survey_link']+=1;continue
        # This route explicitly has no eligible NSC r anchor; it cannot reuse a reserved detection.
        if np.any(nsc[index]&data.band.iloc[index].eq('r').to_numpy()):continue
        counts['non_r_anchor_tracks_before_time_and_quality']+=1
        if any(i in used for i in index):counts['non_r_shared_detection_competition']+=1;continue
        if not valid_epochs(index,nsc,time,prod):counts['non_r_time_exit']+=1;continue
        shape=data.morphology.iloc[index[nsc[index]]].to_numpy(float)
        if not np.isfinite(shape).any() or np.nanmedian(shape)<.8:counts['non_r_morphology_exit']+=1;continue
        model,err=fit_with_calibration(xy[index],time[index],sigma[index],config['reference_epoch'],[solution[k] for k in keys[index]])
        if not core(model.coeff[0]):counts['non_r_ownership_exit']+=1;continue
        if not np.isfinite(err).all() or err.max()>prod['max_axis_pm_error_masyr'] or np.quantile(np.linalg.norm(model.residual/sigma[index],axis=1),.68)>3:
            counts['non_r_fit_quality_exit']+=1;continue
        mag,mag_source=r_magnitude(index,data,nsc)
        if not 20<=mag<23:counts['non_r_r_magnitude_domain_exit']+=1;continue
        candidates.append(dict(kind='non_r_anchor',index=index,magnitude=mag,magnitude_source=mag_source,baseline=None));used.update(index)
    coord=xy-np.array([solution[k][0] if k in solution else np.zeros(2) for k in keys]);training=[];models=[]
    for row in candidates:
        index=row['index']
        if held[index].any():raise ValueError('Target held-year leakage')
        model,_=fit_with_calibration(xy[index],time[index],sigma[index],config['reference_epoch'],[solution[k] for k in keys[index]])
        std=evaluated_fit(index,data,xy,time,sigma,keys,solution,center,config['reference_epoch'])
        item={k:v for k,v in row.items() if k not in ('index','baseline')}
        item.update(detection_ids=data.detection_id.iloc[index].tolist(),training_years=sorted(set(map(int,year[index]))),
            position=model.coeff[0],motion=model.coeff[1],pm_icrs_masyr=std['pm_icrs_masyr'],pm_error_masyr=std['pm_error_masyr'],
            bands=sorted(set(data.band.iloc[index])),exposures=sorted(set(visit[index])))
        training.append(item);models.append((model,index))
    # Persist frozen target identities before constructing any held-target search or QSO lookup.
    save_json(work/'private_training.json',training)
    result=dict(status='completed',year=held_year,reference_gauge=gauge,selection_funnel=dict(counts),r_anchor_funnel=funnel,
        frozen_r_training_solutions=len(baseline),all_filter_r_anchor_training_solutions=len(products),
        train_candidates=len(candidates),slurm_job=os.environ['SLURM_JOB_ID'])
    if not candidates:save_json(work/'private_held.json',[]);save_json(work/'allband_blind_year_summary.json',result);return
    points=np.array([m.coeff[0] for m,_ in models]);motion=np.array([m.coeff[1] for m,_ in models])
    # Small all-filter metadata request, not an unbounded measurement-table scan.
    local=tuple(float(np.mean(spec[k])) for k in ('ra','dec'));radius=float(np.hypot(np.ptp(spec['ra']),np.ptp(spec['dec']))/2+.06)
    chips=access.checked_spatial(access.Cache(work/'footprint_cache'),'nsc',','.join(['exposure','instrument','filter','mjd']+CHIP_VERTICES),'nsc_dr2.chip',access.q3c(*local,radius+.20),local[1]-radius-.200001,local[1]+radius+.200001)
    chips.to_csv(work/'private_allband_chips.csv',index=False)
    utc_year=np.array([int(v[:4]) for v in Time(chips.mjd.to_numpy(),format='mjd',scale='utc').isot]);chips=chips.loc[utc_year==held_year].copy()
    tests=nsc&physical&support&held;coverage={}
    for label,mask in [('all_archived_chips',np.ones(len(chips),bool)),('supported_groups',('NSC:'+chips.exposure.astype(str)).isin(solution).to_numpy()),('retrieved_qualified_exposures',('NSC:'+chips.exposure.astype(str)).isin(set(visit[tests])).to_numpy())]:
        covered=np.zeros(len(candidates),bool)
        for midpoint,polygons in r_chip_shapes(chips.loc[mask].reset_index(drop=True),center):
            pred=points+motion*(midpoint-config['reference_epoch'])
            for polygon in polygons:covered |= polygon.contains_points(pred)
        coverage[label]=covered
    matches=[];null=[set() for _ in range(4)];claims=0
    shifts=np.array([(30000,10000),(-30000,10000),(10000,-30000),(-10000,-30000)])
    for _,frame in data.loc[tests].groupby('exposure_id',sort=False):
        ix=frame.index.to_numpy();midpoint=np.median(time[ix]);tree=cKDTree(coord[ix]);pred=points+motion*(midpoint-config['reference_epoch'])
        good,nearest,collision=exclusive(tree,pred,1000,prod['competitor_margin_mas']);claims+=collision
        for j,shift in enumerate(shifts):
            valid,_,_=exclusive(tree,pred+shift,1000,prod['competitor_margin_mas']);null[j].update(np.flatnonzero(valid))
        for slot in np.flatnonzero(good):
            k=ix[nearest[slot]];model,use=models[slot]
            variance=held_prediction_variance(model,time[use],sigma[use],[solution[g] for g in keys[use]],solution[keys[k]],sigma[k],time[k])
            matches.append(dict(slot=int(slot),band=data.band.iloc[k],held_detection_id=data.detection_id.iloc[k],residual_mas=coord[k]-model.predict(time[k]),error_mas=np.sqrt(variance)))
    save_json(work/'private_held.json',matches)
    cohorts={}
    for kind in ('r_existing','extra_r_anchor','non_r_anchor'):
        for low in (20,21,22):
            slots={i for i,r in enumerate(training) if r['kind']==kind and low<=r['magnitude']<low+1}
            sample=[r for r in matches if r['slot'] in slots];vectors=np.asarray([r['residual_mas'] for r in sample]);errors=np.asarray([r['error_mas'] for r in sample])
            stats=dict(training_sources=len(slots),matched_sources=ratio(len({r['slot'] for r in sample}),len(slots)),matched_positions=len(sample),
                coverage={name:int(sum(mask[j] for j in slots)) for name,mask in coverage.items()},
                shifted_position_matches=[ratio(len(slots&v),len(slots)) for v in null],magnitude_source_counts=dict(Counter(training[j]['magnitude_source'] for j in slots)))
            if len(sample)>=5:stats.update(median_residual_mas=np.median(vectors,axis=0),radial_p68_mas=np.quantile(np.linalg.norm(vectors,axis=1),.68),within_1sigma=(abs(vectors)<=errors).mean(axis=0))
            cohorts[kind+'__r'+str(low)]=stats
    qso=pd.read_csv(source/'spectra_union.csv');found,nearest=query_unique(cKDTree(project_qso(qso,center)),points,500,300)
    claims_qso=Counter(nearest[found]);stationary=defaultdict(list)
    for j in np.flatnonzero(found):
        if claims_qso[nearest[j]]==1:stationary[training[j]['kind']+'__r'+str(int(training[j]['magnitude']))].append(training[j]['pm_icrs_masyr'])
    result.update(by_class=cohorts,spectral_zero={k:dict(n=len(v),median_masyr=np.median(v,axis=0),nmad_masyr=1.4826*np.median(abs(np.asarray(v)-np.median(v,axis=0)),axis=0)) for k,v in stationary.items()},
        shared_held_claims_rejected=claims,
        scope='Whole held year excluded from every target band before association. Frozen r-anchor additions and independent no-NSC-r anchor route separated; non-r r magnitudes come from actual SDSS r and are labelled. All bands eligible by physical quality; object IDs never enter training association. One-to-one 1arcsec held matches and four displaced-position controls. Conditional residuals are not absolute dark-star PM truth.',
        limitations='Candidate counts only, not adopted reliable stars. Unknown identities/blends are not successes; native W0 shared-night errors are uncalibrated. Footprints of qualified exposures do not prove a faint target had sufficient SNR. Original object-index r acquisition selection remains. No adopted full catalogue until independent source/error/zero validation.')
    check_aggregate(result);save_json(work/'allband_blind_year_summary.json',result)


def project_qso(qso,center):
    from .astrometry import project
    return project(qso.ra,qso.dec,center)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('work','source','shared','geometry'):p.add_argument(name,type=Path)
    p.add_argument('region');a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use a Slurm compute node')
    a.work.mkdir(parents=True,exist_ok=True)
    bind(a.work,'source_receipt.json',verify_directory(a.source,('detections.csv','spectra_union.csv')))
    bind(a.work,'reference_receipt.json',verify_directory(a.shared,('private_c0.npz',)))
    bind(a.work,'production_config.toml',SPATIAL);bind(a.work,'geometry_config.toml',a.geometry)
    with run_stage(a.work,PRECISION,a.region,'allband-blind-year-v14',inputs=('source_receipt.json','reference_receipt.json','production_config.toml','geometry_config.toml'),outputs=('allband_blind_year_summary.json','private_training.json','private_held.json')) as go:
        if go:run(a.work,a.source,a.shared,tomllib.loads(a.geometry.read_text())['fields'][a.region])


if __name__=='__main__':main()
