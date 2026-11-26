"""Trace the frozen 415 deep optical candidates through old and sky geometry."""
import argparse
from collections import Counter,defaultdict
from functools import lru_cache
import gzip
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from astropy.time import Time

from . import access
from .__main__ import tomllib
from .allband_qso import evaluated_fit,physical_candidates
from .astrometry import project
from .blind_catalogue_quality import read
from .continuous_reference import PRECISION,SPATIAL,read_offsets
from .coordinate_candidate import metadata_chips,opportunities
from .coordinate_metric import AngularTree
from .dark_precision import query_unique
from .exposure_catalogues import inside
from .io import save_json,json_default,digest
from .ngc5466_demo import TARGET
from .nsc_science_catalogue import select_catalogue
from .pilot_production import quality_detections
from .precision_v03 import calibration_groups,calibration_support,fit_with_calibration
from .public_summary import check_aggregate

ORDER=['pointlike_morphology','no_sdss_position','anchor_competition','time_baseline','m3_support','quality','formal_error','residual','final']


def run(registration,photometry,association,output,cache):
    output.mkdir(parents=True,exist_ok=True);p=read(registration)
    phot=pd.read_csv(photometry);phot=phot.loc[phot.cmd_v20&phot.radius.between(*TARGET,inclusive='left')&phot.r0.between(22,23,inclusive='left')].copy()
    if len(phot)!=415 or sum(phot.r0.lt(22.5))!=212:raise ValueError('Frozen NGC optical denominator changed')
    phot['frozen_index']=phot.index;config=tomllib.loads(PRECISION.read_text());settings=tomllib.loads(SPATIAL.read_text())
    traces=[];seen=set()
    for row in p['selected']:
        if not row['ngc']:continue
        chosen=phot.loc[inside(phot.ra,phot.dec,row['spec'])].copy()
        if chosen.empty:continue
        source=Path(row['source']);center=row['graph_center'];solution,gauges=read_offsets(Path(row['graph']),with_gauge=True)
        solution={g:v for g,v in solution.items() if gauges[g]==row['reference_gauge']}
        data,xy,t,sigma,keep=quality_detections(source,center,config,settings['production']);keys=calibration_groups(data)
        nsc=data.survey.eq('NSC').to_numpy();support=calibration_support(keys,solution);physical=physical_candidates(data)
        goals=project(chosen.ra,chosen.dec,center)
        raw=pd.read_csv(source/'detections.csv',low_memory=False);raw=raw.loc[np.isfinite(raw[['ra_deg','dec_deg']]).all(axis=1)].reset_index(drop=True)
        rawxy=project(raw.ra_deg,raw.dec_deg,center);rawnear=AngularTree(rawxy,center).query_ball_point(goals,1000)
        newnear=AngularTree(xy,center).query_ball_point(goals,1000);oldnear=cKDTree(xy).query_ball_point(goals,1000)
        @lru_cache(maxsize=32768)
        def fit(ix):
            index=np.asarray(ix,int);return fit_with_calibration(xy[index],t[index],sigma[index],config['reference_epoch'],[solution[g] for g in keys[index]])
        def final(ix):return evaluated_fit(ix,data,xy,t,sigma,keys,solution,center,config['reference_epoch'],model=fit(tuple(ix))[0])
        audit={};_,_,_,_=select_catalogue(data,xy,t,sigma,keep,solution,keys,center,row['spec'],config,settings,lambda ix:fit(tuple(ix)),final,selection_audit=audit)
        skyaudit=read(association/row['core']/'private_geometry_selection_trace.json.gz')['B3_spherical']
        old=read(Path(row['fit'])/'private_standard_motion.json.gz');new=read(association/row['core']/'private_B3_spherical_catalogue.json.gz')
        matches={};relations={}
        for name,records,events in [('old',old,audit),('geometry',new,skyaudit)]:
            points=project(*np.asarray([r['position_deg'] for r in records]).T,center)
            good,near=query_unique(AngularTree(points,center),goals,700,300);matches[name]=dict(zip(np.flatnonzero(good),near[good]))
            rel=defaultdict(list)
            for route in ('all_r_anchor','non_r'):
                for event in events[route]:
                    if event['status'] in ('outside_core','contains_r','missing_survey'):continue
                    for i in event.get('anchor_indices',event.get('track_indices',[])):
                        if nsc[i]:rel[int(i)].append(event)
            relations[name]=rel
        chips,meta=metadata_chips(source,access.Cache(cache/row['core']))
        # Opportunity for this fixed optical sample is evaluated in every epoch,
        # rather than pretending its missing PM is zero truth. Over this small
        # field a 0-motion footprint proxy has subarcsec boundary limitations.
        models=[SimpleNamespace(coeff=np.array([goal,[0.,0.]]),epoch=2015.5) for goal in goals]
        chipyears=np.unique(np.asarray(Time(chips.mjd.dropna(),format='mjd',scale='utc').datetime64,dtype='datetime64[Y]').astype(int)+1970)
        opp={k:np.zeros(len(chosen),bool) for k in ('geometric','catalogue_10sigma','background_limited_5sigma_proxy','depth_or_band_unknown')}
        for year in chipyears:
            values,_=opportunities(raw,models,[np.asarray(v,int) for v in rawnear],chips,center,solution,int(year))
            for key in opp:opp[key]|=values[key]
        for j,(_,target) in enumerate(chosen.iterrows()):
            identifier=int(target.frozen_index)
            if identifier in seen:raise ValueError('Optical source assigned twice')
            seen.add(identifier);near=np.asarray(rawnear[j],int);counts=Counter(raw.survey.iloc[near]);statuses={};overlap=[]
            for name,nearby in [('old',oldnear[j]),('geometry',newnear[j])]:
                ix=np.asarray(nearby,int);early=ix[~nsc[ix]&keep[ix]];late=ix[nsc[ix]&physical[ix]]
                options={id(e):e for i in late for e in relations[name][int(i)]}
                if j in matches[name]:status='recovered'
                elif counts['NSC']==0:status='no_published_NSC_detection'
                elif counts['SDSS']==0:status='no_published_SDSS_detection'
                elif not len(late):status='NSC_physical_quality_or_distance'
                elif not len(early):status='SDSS_quality_or_distance'
                elif not options:status='no_eligible_anchor'
                else:
                    status=max(options.values(),key=lambda e:ORDER.index(e['status']) if e['status'] in ORDER else -1)['status']
                    if status=='final':status='identity_or_owner'
                statuses[name]=dict(primary_exit=status,qualified_SDSS=len(early),physical_NSC=len(late),
                    supported_NSC=int(support[late].sum()),matching_NSCr=int(data.band.iloc[late].eq('r').sum()),
                    competing_tracks=len(options),all_anchor_statuses=dict(Counter(e['status'] for e in options.values())))
            if len(newnear[j])>len(oldnear[j]):overlap.append('sky_radius_adds_published_detection')
            if opp['geometric'][j] and not counts['NSC']:overlap.append('polygon_opportunity_without_retrieved_NSC')
            traces.append(dict(frozen_index=identifier,r0=float(target.r0),core=row['core'],statuses=statuses,
                raw_SDSS=int(counts['SDSS']),raw_NSC=int(counts['NSC']),opportunity={k:bool(v[j]) for k,v in opp.items()},overlap_flags=overlap))
        print('NGC_DEEP_TRACE',row['core'],len(chosen),flush=True)
    if len(seen)!=415:raise ValueError('Frozen optical targets missing from registered cores')
    results={}
    for lo in (22.,22.5):
        subset=[r for r in traces if lo<=r['r0']<lo+.5]
        results[str(lo)]=dict(optical_candidates=len(subset),models={name:dict(Counter(r['statuses'][name]['primary_exit'] for r in subset)) for name in ('old','geometry')},
            opportunities={key:sum(r['opportunity'][key] for r in subset) for key in opp},
            overlap_flags=dict(Counter(flag for r in subset for flag in r['overlap_flags'])))
        for counts in results[str(lo)]['models'].values():assert sum(counts.values())==len(subset)
    with gzip.open(output/'private_ngc415_trace.json.gz','wt') as f:json.dump(traces,f,default=json_default,allow_nan=False)
    report=dict(photometry_sha256=digest(photometry),fixed_candidates=415,bins=results,slurm_job=os.environ['SLURM_JOB_ID'],
        scope='Frozen independent optical CMD:212+203 candidates, not certified cluster members. Same r0 optical denominator and original0.8 shape. Old vs full2D/sky geometry conditional on old C0; unique static optical match0.7arcsec/0.3arcsec. Raw input counts use physical1arcsec. Actual CCD polygon/reference/depth opportunities, brightness from retrieved same-band data: missing bands and internal effective-pixel masks stay unknown. Static footprint proxy cannot certify high motion or subarcsec edges. No image extraction completeness claim.')
    check_aggregate(report);save_json(output/'v21_ngc415_trace_summary.json',report)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('registration','photometry','association','output','cache'):p.add_argument(n,type=Path)
    a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use Slurm resources')
    run(**vars(a))
