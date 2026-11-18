"""Whole-year r-only position prediction with the final regional C0 graph.

Training associations are built afresh, before any held detections are searched.
This tests conditional position predictions, not dark-star absolute PM truth.
"""
import argparse
from collections import Counter
import os
from pathlib import Path

from astropy.time import Time
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .__main__ import tomllib
from .access import Cache
from .astrometry import unproject
from .condition_holdout import held_prediction_variance
from .continuous_reference import FILES,PRECISION,SPATIAL,bind,read_offsets,reference_frame,verify_directory
from .dark_precision import query_unique
from .exposure_catalogues import inside
from .io import save_json
from .pilot_production import make_tracks,quality_detections,source_funnel
from .precision_v03 import calibration_groups,calibration_support
from .provenance import run_stage
from .public_summary import check_aggregate
from .regional_preflight import r_chips
from .regional_production import r_chip_shapes
from .validation import ratio


def choose_year(time,exposures,nsc,eligible):
    unique,inverse=np.unique(time,return_inverse=True)
    year=np.array([int(v[:4]) for v in Time(unique,format='jyear',scale='tt').utc.isot])[inverse]
    years=np.unique(year[eligible])
    if len(years)<3:return year,None
    chosen=max(years[1:-1],key=lambda y:len(set(exposures[eligible&(year==y)])))
    return year,int(chosen)


def exclusive(tree,predictions,radius,margin):
    distance,_=tree.query(predictions,k=2)
    candidates=(distance[:,0]<=radius)&(distance[:,1]-distance[:,0]>margin)
    good,nearest=query_unique(tree,predictions,radius,margin)
    # query_unique already excludes all many-to-one claims; count those exits.
    return good,nearest,int(candidates.sum()-good.sum())


def predict(work,data,xy,time,sigma,keys,nsc,keep,offsets,settings,config,spec,center,chip_shapes):
    calibrated=calibration_support(keys,offsets)
    eligible=nsc&keep&calibrated
    exposures=data.exposure_id.to_numpy()
    year,held_year=choose_year(time,exposures,nsc,eligible)
    if held_year is None:
        return dict(status='insufficient_calibrated_nsc_years',n_years=len(set(year[eligible])))
    held=nsc&(year==held_year)
    train=keep&~held
    prod=settings['production']
    tracks=make_tracks(xy,exposures,train,prod['match_radius_mas'])
    seeds=make_tracks(xy,exposures,train&nsc,prod['match_radius_mas'])
    core=lambda position:inside(*unproject(position,center)[0],spec)
    fits={}
    _,products=source_funnel(data,xy,time,sigma,train,tracks,offsets,config,settings,9.,
        initial=seeds,observation_groups=keys,core_members=core,fits=fits)
    slots=[j for j,v in products.items() if v[3] in ('20_to_21','21_to_22','22_to_23') and core(v[0])]
    training=[]
    for j in slots:
        use=products[j][4]
        if np.any(held[use]) or not np.all(train[use]):raise ValueError('Held-year training leakage')
        training.append(dict(track_id=j,r_bin=products[j][3],training_detection_ids=data.detection_id.iloc[use].tolist(),
            training_utc_years=sorted(set(map(int,year[use]))),position=fits[j].coeff[0],motion=fits[j].coeff[1]))
    # Persist before constructing the held-position search.
    save_json(work/'private_training.json',training)
    if not slots:return dict(status='no_eligible_training_sources',year=held_year)
    positions=np.array([fits[j].coeff[0] for j in slots]);motions=np.array([fits[j].coeff[1] for j in slots])
    covered=np.zeros(len(slots),bool);near=np.zeros(len(slots),bool)
    for midpoint,polygons in chip_shapes:
        if int(Time(midpoint,format='jyear',scale='tt').utc.isot[:4])!=held_year:continue
        pred=positions+motions*(midpoint-config['reference_epoch'])
        for polygon in polygons:covered |= polygon.contains_points(pred)
    matched=[];null=[set() for _ in range(4)];collisions=0;visits=0
    shifts=np.array([(30000.,10000.),(-30000.,10000.),(10000.,-30000.),(-10000.,30000.)])
    for _,frame in data.loc[eligible&held].groupby('exposure_id',sort=False):
        ix=frame.index.to_numpy();midpoint=np.median(time[ix]);visits+=1
        coordinates=xy[ix]-np.array([offsets[k][0] for k in keys[ix]])
        tree=cKDTree(coordinates);pred=positions+motions*(midpoint-config['reference_epoch'])
        near |= tree.query(pred)[0]<=1000
        good,nearest,collision=exclusive(tree,pred,1000,prod['competitor_margin_mas']);collisions+=collision
        for index,shift in enumerate(shifts):
            accepted,_,_=exclusive(tree,pred+shift,1000,prod['competitor_margin_mas'])
            null[index].update(np.flatnonzero(accepted))
        for local in np.flatnonzero(good):
            j=slots[local];k=ix[nearest[local]];model=fits[j];use=products[j][4]
            residual=coordinates[nearest[local]]-model.predict(time[k])
            variance=held_prediction_variance(model,time[use],sigma[use],[offsets[keys[i]] for i in use],
                offsets[keys[k]],sigma[k],time[k])
            matched.append(dict(track_id=j,r_bin=products[j][3],held_detection_id=data.detection_id.iloc[k],
                residual_mas=residual,error_mas=np.sqrt(variance)))
    save_json(work/'private_held_predictions.json',matched)
    cohorts={}
    for tag in ('20_to_21','21_to_22','22_to_23'):
        members=np.array([i for i,j in enumerate(slots) if products[j][3]==tag],int)
        selected=[r for r in matched if r['r_bin']==tag];found={r['track_id'] for r in selected}
        out=dict(train_sources=len(members),matched_sources=ratio(len(found),len(members)),matched_positions=len(selected),
            chip_covered_sources=int(covered[members].sum()),near_held_detection_sources=int(near[members].sum()),
            covered_without_near_detection=int((covered&~near)[members].sum()),
            geometric_or_archive_unknown=int((~covered)[members].sum()),
            shifted_position_matches=[ratio(len(set(members)&v),len(members)) for v in null])
        if len(selected)>=5:
            residual=np.array([r['residual_mas'] for r in selected]);error=np.array([r['error_mas'] for r in selected])
            median=np.median(residual,axis=0)
            out.update(radial_p68_mas=np.quantile(np.linalg.norm(residual,axis=1),.68),median_mas=median,
                nmad_mas=1.4826*np.median(abs(residual-median),axis=0),
                within_1sigma_fraction=np.mean(abs(residual)<=error,axis=0),
                within_2sigma_fraction=np.mean(abs(residual)<=2*error,axis=0),
                beyond_3sigma_fraction=np.mean(abs(residual)>3*error,axis=0))
        cohorts[tag]=out
    return dict(status='completed',year=held_year,held_exposure_count=visits,shared_detection_claims_rejected=collisions,
        by_r=cohorts,training_has_no_held_year_detections=True,
        scope='Frozen C0/W0 r-only; training-only association; 1 arcsec prediction gate, unique competing detections and exclusive claims. Residual and formal coverage conditional on that gate; shifted positions estimate held-position chance matches only. Chip polygons identify geometric opportunities, not image-level completeness. Reference galaxies may calibrate held exposures; no target held detections enter training.')


def run(work,region,source,shared,geometry):
    config=tomllib.loads(PRECISION.read_text());settings=tomllib.loads(SPATIAL.read_text())
    spec=tomllib.loads(geometry.read_text())['fields'][region]
    graph=shared/'private_c0.npz';center=reference_frame(graph);offsets,gauges=read_offsets(graph,with_gauge=True)
    data,xy,time,sigma,keep=quality_detections(source,center,config,settings['production'])
    keys=calibration_groups(data);nsc=data.survey.eq('NSC').to_numpy()
    counts=pd.Series(keys[keep&inside(data.ra_deg,data.dec_deg,spec)]).map(gauges).value_counts()
    main=min(counts.index,key=lambda k:(-counts[k],k)) if len(counts) else None
    offsets={k:v for k,v in offsets.items() if gauges[k]==main}
    local_center=tuple(float(np.mean(spec[k])) for k in ('ra','dec'))
    radius=float(np.hypot(np.ptp(spec['ra']),np.ptp(spec['dec']))/2+.06)
    chips=r_chip_shapes(r_chips(Cache(work/'footprint_cache'),local_center,radius),center)
    result=predict(work,data,xy,time,sigma,keys,nsc,keep,offsets,settings,config,spec,center,chips)
    result.update(reference_gauge=main,slurm_job=os.environ['SLURM_JOB_ID'])
    check_aggregate(result);save_json(work/'year_prediction_summary.json',result)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('work','region','source','shared','geometry'):parser.add_argument(name,type=str if name=='region' else Path)
    a=parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):parser.error('Use a Slurm compute node')
    a.work.mkdir(parents=True,exist_ok=True)
    bind(a.work,'source_receipt.json',verify_directory(a.source,FILES))
    bind(a.work,'reference_receipt.json',verify_directory(a.shared,('private_c0.npz',)))
    bind(a.work,'production_config.toml',SPATIAL);bind(a.work,'geometry_config.toml',a.geometry)
    with run_stage(a.work,PRECISION,a.region,'independent-year-v13',
        inputs=('source_receipt.json','reference_receipt.json','production_config.toml','geometry_config.toml'),
        outputs=('year_prediction_summary.json',)) as go:
        if go:run(a.work,a.region,a.source,a.shared,a.geometry)


if __name__=='__main__':main()
