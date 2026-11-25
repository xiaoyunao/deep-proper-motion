"""One registered morphology candidate, with whole-year target isolation."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import gzip
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from astropy.time import Time

from .__main__ import tomllib
from .allband_qso import evaluated_fit,physical_candidates
from .astrometry import project
from .blind_catalogue_quality import read
from .continuous_reference import PRECISION,SPATIAL,read_offsets,reference_frame
from .dark_precision import query_unique
from .exposure_catalogues import inside
from .independent_year import exclusive
from .io import digest,save_json,json_default
from .nsc_science_catalogue import select_catalogue
from .pilot_production import make_tracks,quality_detections
from .precision_v03 import calibration_groups,calibration_support,fit_with_calibration
from .public_summary import check_aggregate
from .selection_semantics import morphology_state
from .usability_cluster import summarize
from .validation import ratio


def independent_acceptor(data,xy,keep,radius=500,margin=300):
    """Run detections first become source tracks; repeated runs are not neighbours."""
    early=keep&data.survey.eq('SDSS').to_numpy()
    tracks=make_tracks(xy,data.exposure_id.to_numpy(),early,500)
    points=np.asarray([np.median(xy[ix],axis=0) for ix in tracks]).reshape(-1,2)
    tree=cKDTree(points)
    star=np.array([data.morphology.iloc[ix].notna().all() and data.morphology.iloc[ix].eq(6).all() for ix in tracks],bool)
    def accept(anchor,position):
        m=morphology_state(data.morphology.iloc[anchor])
        if m['valid_count']!=len(anchor) or m['valid_median'] is None or not .6<=m['valid_median']<.8:return False
        valid,index=query_unique(tree,np.asarray(position).reshape(1,2),radius,margin)
        return bool(valid[0] and star[index[0]])
    return accept


def one(task):
    row,output=task;dest=Path(output)/row['core'];dest.mkdir(parents=True,exist_ok=True)
    result=dest/'v20_selection_candidate_core_summary.json'
    ticket=dict(source_sha256=digest(Path(row['source'])/'run_manifest.json'),
        graph_sha256=digest(Path(row['graph'])/'private_c0.npz'),implementation_sha256=digest(Path(__file__)))
    if result.exists():
        prior=read(result)
        if prior['ticket']!=ticket:raise ValueError('Candidate input or implementation changed')
        return prior
    config=tomllib.loads(PRECISION.read_text());settings=tomllib.loads(SPATIAL.read_text())
    spec=tomllib.loads(Path(row['geometry']).read_text())['fields'][row['core']]
    graph=Path(row['graph'])/'private_c0.npz';center=reference_frame(graph)
    solution,gauges=read_offsets(graph,with_gauge=True)
    solution={k:v for k,v in solution.items() if gauges[k]==row['reference_gauge']}
    data,xy,t,sigma,keep=quality_detections(Path(row['source']),center,config,settings['production'])
    keys=calibration_groups(data);nsc=data.survey.eq('NSC').to_numpy()
    support=calibration_support(keys,solution);physical=physical_candidates(data)
    years=np.asarray(Time(t,format='jyear',scale='tt').utc.datetime64,dtype='datetime64[Y]').astype(int)+1970
    held_year=int(np.max(years[nsc&physical&support]));held=nsc&(years==held_year)
    frozen=read(Path(row['fit'])/'private_standard_motion.json.gz')
    baseline={tuple(sorted(r['detection_ids'])):r for r in frozen}
    samples={};private={};full_new_qso=[]
    spectra=pd.read_csv(Path(row['source'])/'spectra_union.csv')
    spectra['original_csv_row']=np.arange(len(spectra))
    spectra=spectra.loc[inside(spectra.ra,spectra.dec,spec)].reset_index(drop=True)
    targets=project(spectra.ra,spectra.dec,center)
    with gzip.open(dest/'private_corrected_spectral_identity_map.json.gz','wt') as stream:
        json.dump(dict(original_table_sha256=digest(Path(row['source'])/'spectra_union.csv'),
            scope='Original CSV row restored before spatial filtering. Archived floating target labels are auxiliary, never certified native IDs.',
            rows=[dict(local_index=int(i),original_csv_row=int(r.original_csv_row),origin=str(r.origin),ra=float(r.ra),dec=float(r.dec)) for i,r in spectra.iterrows()]),stream,allow_nan=False)
    for phase,retain in [('full',np.ones(len(data),bool)),('training',~held)]:
        # Slice every array: a keep mask alone lets select_catalogue re-add held bands.
        d=data.loc[retain].reset_index(drop=True);x=xy[retain];time=t[retain];err=sigma[retain];k=keys[retain];rkeep=keep[retain]
        if phase=='training' and np.any(d.survey.eq('NSC')&(years[retain]==held_year)):raise ValueError('Held target leaked')
        @lru_cache(maxsize=32768)
        def solve(ix):
            index=np.asarray(ix,int)
            return fit_with_calibration(x[index],time[index],err[index],config['reference_epoch'],[solution[g] for g in k[index]])
        def fit(index):return solve(tuple(index))
        def final(index):return evaluated_fit(index,d,x,time,err,k,solution,center,config['reference_epoch'],model=fit(index)[0])
        accept=independent_acceptor(d,x,rkeep)
        _,candidate,_,_=select_catalogue(d,x,time,err,rkeep,solution,k,center,spec,config,settings,fit,final,morphology_accept=accept)
        mapped={tuple(sorted(d.detection_id.iloc[v['index']])):v for v in candidate}
        if phase=='full':base=baseline
        else:
            _,original,_,_=select_catalogue(d,x,time,err,rkeep,solution,k,center,spec,config,settings,fit,final)
            base={tuple(sorted(d.detection_id.iloc[v['index']])):v for v in original}
        new=set(mapped)-set(base);old=set(mapped)&set(base)
        samples[phase]=dict(original=len(base),candidate=len(mapped),identical_members_retained=len(old),
            retention=ratio(len(old),len(base)),new=len(new),held_year=held_year)
        # Freeze new identities before opening held targets or evaluating QSO motion.
        private[phase]=[dict(detection_ids=list(member),kind=mapped[member]['kind']) for member in sorted(new)]
        with gzip.open(dest/f'private_{phase}_candidate_members.json.gz','wt') as stream:json.dump(private[phase],stream,default=json_default,allow_nan=False)
        qso=[];hold_models=[]
        for member in sorted(new):
            index=mapped[member]['index'];model,_=fit(index)
            if phase=='training':hold_models.append(model)
            good,near=query_unique(cKDTree(targets),model.coeff[0].reshape(1,2),500,300)
            if good[0]:
                j=int(near[0]);qso.append(dict(identity=j,magnitude=float(spectra.reference_r.iat[j]),motion=final(index)['pm_icrs_masyr']))
        claims=Counter(r['identity'] for r in qso)
        qso=[r for r in qso if claims[r['identity']]==1]
        samples[phase]['added_qso']={str(low):summarize([r['motion'] for r in qso if low<=r['magnitude']<low+1]) for low in (21,22)}
        if phase=='full':full_new_qso=qso
        else:
            matched=set();null=[set() for _ in range(4)];residual=[];collisions=0
            shifts=[(30000,10000),(-30000,10000),(10000,-30000),(-10000,-30000)]
            for _,frame in data.loc[held&physical&support].groupby('exposure_id',sort=False):
                ix=frame.index.to_numpy();mid=np.median(t[ix]);coord=xy[ix]-np.array([solution[g][0] for g in keys[ix]])
                pred=np.asarray([model.predict(mid) for model in hold_models]).reshape(-1,2)
                tree=cKDTree(coord)
                found,nearest,collision=exclusive(tree,pred,1000,settings['production']['competitor_margin_mas']);collisions+=collision
                for j in np.flatnonzero(found):
                    matched.add(int(j));residual.append(coord[nearest[j]]-hold_models[j].predict(t[ix[nearest[j]]]))
                for i,shift in enumerate(shifts):
                    found,_,_=exclusive(tree,pred+shift,1000,settings['production']['competitor_margin_mas']);null[i].update(np.flatnonzero(found))
            samples[phase]['held_year']=dict(year=held_year,training_new_sources=len(new),uniquely_reproduced=len(matched),
                conditional_ratio=ratio(len(matched),len(new)),shifted_counts=list(map(len,null)),
                competing_claims_rejected=collisions,position_residual=summarize(residual),
                scope='All held target bands excluded before association. One-to-one 1arcsec and four shifts. Denominator is training candidates, not independently verified CCD/SNR opportunities or true stars.')
    report=dict(ticket=ticket,core=row['core'],samples=samples,slurm_job=os.environ['SLURM_JOB_ID'])
    check_aggregate(report);save_json(result,report);print('CANDIDATE_CORE_DONE',row['core'],samples['full']['new'],flush=True)
    return report


def run(registration,candidate_registration,output,workers):
    output.mkdir(parents=True,exist_ok=True);plan=read(registration);registered=read(candidate_registration)
    with ProcessPoolExecutor(max_workers=workers) as pool:cores=list(pool.map(one,[(r,str(output)) for r in plan['selected']]))
    counts={phase:{name:sum(c['samples'][phase][name] for c in cores) for name in ('original','candidate','identical_members_retained','new')} for phase in ('full','training')}
    # Exact retained members are fitted with identical inputs/weights: no point-estimate change.
    # Certification additionally requires enough independent added identities and morphology truth;
    # a reproduction test alone is deliberately insufficient for stellar adoption.
    report=dict(registration_sha256=digest(candidate_registration),cores=cores,counts=counts,
        candidate_adopted=False,decision='Close the one candidate for adopted dark-star production: these tests do not provide an independent extended-source/blend purity denominator or full CCD/SNR opportunity audit. Keep measured recovery and QSO zero results as conditional evidence; do not tune a second threshold.',
        scope='Single registered 0.6 morphology relaxation requiring unique SDSS STAR source. Baseline fits and unknown-morphology policy unchanged. No production catalogue overwritten.',slurm_job=os.environ['SLURM_JOB_ID'])
    check_aggregate(report);save_json(output/'v20_selection_candidate_summary.json',report)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('registration','candidate_registration','output'):p.add_argument(n,type=Path)
    p.add_argument('--workers',type=int,default=2);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use Slurm compute resources')
    run(**vars(a))
