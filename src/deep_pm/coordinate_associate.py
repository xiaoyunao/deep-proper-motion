"""B3 actual angular gates with original morphology and frozen shared C0."""
import argparse
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import gzip
import json
import os
from pathlib import Path
import resource
from time import monotonic

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .__main__ import tomllib
from .allband_qso import physical_candidates
from .astrometry import project
from .blind_catalogue_quality import read
from .continuous_reference import PRECISION,SPATIAL,read_offsets,verify_directory
from .coordinate_metric import (native_covariance,projected_covariance,AngularTree,
    fit_vector_motion,reference_covariance)
from .coordinate_replay import quantile
from .dark_precision import query_unique
from .exposure_catalogues import inside
from .io import digest,save_json,json_default
from .nsc_science_catalogue import select_catalogue
from .pilot_production import quality_detections
from .precision_v03 import calibration_groups,calibration_support,reference_rows
from .public_summary import check_aggregate
from .standard_motion import icrs_motion


def member_key(row):return tuple(sorted(map(str,row['detection_ids'])))


def edge_counts(xy,early,late,center):
    indices=np.flatnonzero(late);old=cKDTree(xy[indices]);new=AngularTree(xy[indices],center)
    result=Counter()
    for start in range(0,int(early.sum()),1024):
        goal=xy[np.flatnonzero(early)[start:start+1024]]
        a=old.query_ball_point(goal,1000);b=new.query_ball_point(goal,1000)
        for left,right in zip(a,b):
            l,r=set(left),set(right)
            result['old_sdss_nsc_r_edges']+=len(l);result['sky_sdss_nsc_r_edges']+=len(r)
            result['new_edges']+=len(r-l);result['lost_edges']+=len(l-r)
    return dict(result)


def one(task):
    row,destination=task;dest=Path(destination)/row['core'];dest.mkdir(parents=True,exist_ok=True)
    source=Path(row['source']);folder=Path(row['fit']);center=row['graph_center']
    ticket=dict(source_sha256=digest(source/'run_manifest.json'),graph_sha256=digest(Path(row['graph'])),
        implementation_sha256=digest(Path(__file__)),metric_sha256=digest(Path(__file__).with_name('coordinate_metric.py')),
        selection_sha256=digest(Path(__file__).with_name('nsc_science_catalogue.py')))
    result=dest/'v21_coordinate_association_core_summary.json'
    if result.exists():
        previous=read(result)
        if previous['ticket']!=ticket:raise ValueError('Association checkpoint signature changed')
        return previous
    start=monotonic();verify_directory(source,('detections.csv','spectra_union.csv'))
    config=tomllib.loads(PRECISION.read_text());settings=tomllib.loads(SPATIAL.read_text())
    data,xy,time,sigma,keep=quality_detections(source,center,config,settings['production'])
    cp=projected_covariance(data.ra_deg,data.dec_deg,center,native_covariance(sigma));inverse=np.linalg.inv(cp)
    groups=calibration_groups(data);nsc=data.survey.eq('NSC').to_numpy()
    offsets,gauges=read_offsets(Path(row['graph']),with_gauge=True)
    solution={g:v for g,v in offsets.items() if gauges[g]==row['reference_gauge']}
    support=calibration_support(groups,solution)
    @lru_cache(maxsize=32768)
    def solve(members):
        ix=np.asarray(members,int);zero=np.asarray([solution[g][0] for g in groups[ix]])
        model=fit_vector_motion(xy[ix]-zero,time[ix],cp[ix],config['reference_epoch'])
        common,draws=reference_covariance(model.influence,np.stack([solution[g][1] for g in groups[ix]],axis=1))
        fit=icrs_motion(model.coeff[0],model.coeff[1],center,model.cov+common)
        fit.update(tangent_state=model.coeff.T.ravel(),reference_tangent_covariance=common,
            independent_tangent_covariance=model.cov,robust_weight=model.weight,span_years=float(np.ptp(time[ix])))
        return model,fit
    def fit_source(ix):
        model,fit=solve(tuple(ix));return model,fit['pm_error_masyr']
    def final(ix):return solve(tuple(ix))[1]
    def distance(ix,model):return np.sqrt(np.einsum('ni,nij,nj->n',model.residual,inverse[ix],model.residual))
    outputs={};audits={};funnel={}
    for tag,sky in [('B2_selected',None),('B3_spherical',center)]:
        audit={}
        _,candidates,f,mask=select_catalogue(data,xy,time,sigma,keep,solution,groups,center,row['spec'],config,settings,fit_source,final,
            selection_audit=audit,sky_center=sky,quality_distance=distance)
        values=[]
        for candidate in candidates:
            ix=candidate['index'];fit=final(ix)
            if not inside(*fit['position_deg'],row['spec']):continue
            values.append(dict(kind=candidate['kind'],detection_ids=data.detection_id.iloc[ix].tolist(),**fit))
        outputs[tag]=values;audits[tag]=audit;funnel[tag]=f
    frozen=read(folder/'private_standard_motion.json.gz');allkeys={tag:{member_key(v) for v in values} for tag,values in dict(old=frozen,**outputs).items()}
    members={}
    for tag,keys in allkeys.items():
        members[tag]=dict(sources=len(keys),same_physical_members_as_old=len(keys&allkeys['old']),
            new_member_sets=len(keys-allkeys['old']),missing_old_member_sets=len(allkeys['old']-keys))
    spectra=pd.read_csv(source/'spectra_union.csv');spectra=spectra.loc[inside(spectra.ra,spectra.dec,row['spec'])].reset_index(drop=True)
    goals=project(spectra.ra,spectra.dec,center);tree=AngularTree(goals,center)
    known_old={r['qso_local_index']:r for r in read(folder/'private_blind_qso.json')}
    matches={};private=[]
    for tag,values in dict(old=frozen,**outputs).items():
        points=project(*np.asarray([r['position_deg'] for r in values]).T,center) if values else np.empty((0,2))
        found,near=query_unique(tree,points,500,300)
        matches[tag]={int(near[j]):values[j] for j in np.flatnonzero(found)}
    for k in sorted(set.union(*(set(v) for v in matches.values()))):
        target=spectra.iloc[k];mag=float(target.reference_r)
        if not 20<=mag<23:continue
        values={tag:(r[k]['pm_icrs_masyr'],r[k]['pm_error_masyr']) for tag,r in matches.items() if k in r}
        if k in known_old:values.update({m:v for m,v in known_old[k]['values'].items() if m in ('NSC','Xiao')})
        private.append(dict(identity=(str(target.origin),float(target.ra),float(target.dec)),magnitude=mag,
            qso_local_index=k,core=row['core'],window=row['window'],values=values))
    # Reference-selection gates are independently measured, without treating
    # newly selected galaxies as an already recalibrated global reference graph.
    labels=pd.read_csv(source/'external_labels.csv',dtype=str).set_index('detection_id').original_object_id
    veto=pd.concat([pd.read_csv(source/name,usecols=['ra','dec']) for name in ('spectra_veto.csv','validation/gaia.csv')],ignore_index=True).dropna()
    ref={}
    for tag,sky in [('old_planar',None),('spherical',center)]:
        selected=reference_rows(data,xy,data.detection_id.map(labels),project(veto.ra,veto.dec,center),config['calibration'],config['seed'],sky_center=sky)
        selected.to_csv(dest/f'private_reference_{tag}.csv.gz',index=False)
        ref[tag]=dict(measurements=len(selected),reference_sources=selected.reference.nunique(),
            training_sources=selected.loc[~selected.is_holdout].reference.nunique(),holdout_sources=selected.loc[selected.is_holdout].reference.nunique())
    edges=edge_counts(xy,keep&~nsc,keep&nsc&physical_candidates(data)&support,center)
    for name,values in outputs.items():
        with gzip.open(dest/f'private_{name}_catalogue.json.gz','wt') as stream:json.dump(values,stream,default=json_default,allow_nan=False)
    with gzip.open(dest/'private_geometry_selection_trace.json.gz','wt') as stream:json.dump(audits,stream,default=json_default,allow_nan=False)
    save_json(dest/'private_geometry_qso.json',private)
    report=dict(ticket=ticket,core=row['core'],center_theta_deg=row['center_theta_deg'],members=members,funnel=funnel,
        candidate_edges=edges,reference_selection=ref,
        qso_counts={tag:{str(low):sum(low<=spectra.reference_r.iat[k]<low+1 for k in table) for low in (21,22)} for tag,table in matches.items()},
        old_planar_label_not_unique_in_sky=sum(k not in matches['old'] for k in known_old),
        elapsed_seconds=monotonic()-start,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        slurm_job=os.environ['SLURM_JOB_ID'],scope='Original0.8 morphology. B2_selected changes weights/quality with old angular approximation; B3 also uses true spherical distances including QSO and reference input gates. Same frozen C0 and gauge. Source count is registered origin-core scientific replay, not a new all-area production claim. Reference input changes need a separately solved graph.')
    check_aggregate(report);save_json(result,report);print('GEOMETRY_ASSOCIATED',row['core'],members,flush=True)
    return report


def run(registration,output,workers,pilot):
    plan=read(registration);selected=[r for r in plan['selected'] if not pilot or r['core'] in plan['pilot']]
    output.mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:results=list(pool.map(one,[(r,str(output)) for r in selected],chunksize=1))
    report=dict(plan_sha256=digest(registration),pilot=pilot,cores=results,slurm_job=os.environ['SLURM_JOB_ID'])
    check_aggregate(report);save_json(output/('v21_association_pilot_summary.json' if pilot else 'v21_association_summary.json'),report)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('registration',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--workers',type=int,default=2);p.add_argument('--pilot',action='store_true');a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use Slurm resources')
    run(**vars(a))
