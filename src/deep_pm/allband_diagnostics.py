"""Fixed-QSO band/year deletion and residual attribution; no QSO coefficient training."""
import argparse
from collections import defaultdict
import gzip
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.time import Time

from .__main__ import tomllib
from .allband_qso import evaluated_fit
from .continuous_reference import PRECISION,SPATIAL,bind,read_offsets,reference_frame,verify_directory
from .io import save_json
from .nsc_comparison import comparison
from .pilot_production import quality_detections,valid_epochs
from .precision_v03 import calibration_groups
from .provenance import run_stage
from .public_summary import check_aggregate


def run(work,pilot):
    with gzip.open(pilot/'private_fixed_qso.json.gz','rt') as stream:records=json.load(stream)
    source=(pilot/'source_receipt.json').resolve().parent
    graph=(pilot/'new_reference_receipt.json').resolve().parent/'private_c0.npz'
    verify_directory(source,('detections.csv',));verify_directory(graph.parent,('private_c0.npz',))
    config=tomllib.loads(PRECISION.read_text());settings=tomllib.loads(SPATIAL.read_text())
    center=reference_frame(graph);solution=read_offsets(graph)
    data,xy,time,sigma,keep=quality_detections(source,center,config,settings['production'])
    keys=calibration_groups(data);lookup=pd.Index(data.detection_id);nsc=data.survey.eq('NSC').to_numpy()
    utc=Time(np.unique(data.observation_time),format='mjd',scale='tt').utc
    years_by_mjd=dict(zip(np.unique(data.observation_time),[int(s[:4]) for s in utc.isot]))
    year=data.observation_time.map(years_by_mjd).to_numpy()
    family=(data.survey+':'+data.band.astype(str)).to_numpy()
    deleted=defaultdict(list);residuals=defaultdict(list);sampling=defaultdict(list);insufficient=defaultdict(int)
    private=[]
    for row in records:
        if 'all_nsc_w0' not in row['fits']:continue
        index=lookup.get_indexer(row['members']['all_nsc_w0'])
        if (index<0).any():raise ValueError('Frozen multiband detections missing')
        baseline=evaluated_fit(index,data,xy,time,sigma,keys,solution,center,config['reference_epoch'])
        np.testing.assert_allclose(baseline['pm_icrs_masyr'],row['fits']['all_nsc_w0']['pm_icrs_masyr'],atol=1e-8,rtol=1e-9)
        state=np.asarray(baseline['tangent_state']);predicted=state[[0,2]]+(time[index,None]-config['reference_epoch'])*state[[1,3]]
        corrected=xy[index]-np.array([solution[k][0] for k in keys[index]])
        delta=corrected-predicted
        part='r'+str(int(row['magnitude']))
        for label in sorted(set(family[index])):
            use=family[index]==label
            # One median per physical validation object, never count its exposures as independent sources.
            residuals[part+'__'+label].append(np.median(delta[use],axis=0))
            ix=index[use]
            times=Time(data.observation_time.iloc[ix].to_numpy(),format='mjd',scale='tt').utc
            sampling[part+'__'+label].append([len(ix),len(set(np.floor(times.mjd-.75).astype(int))),len(set(year[ix])),np.ptp(time[ix])])
        leave={}
        alternatives={'without_band_'+b: index[~(nsc[index]&data.band.iloc[index].eq(b).to_numpy())] for b in sorted(set(data.band.iloc[index[nsc[index]]]))}
        alternatives.update({'without_nsc_year_'+str(y):index[~(nsc[index]&(year[index]==y))] for y in sorted(set(year[index[nsc[index]]]))})
        for name,left in alternatives.items():
            if not valid_epochs(left,nsc,time,settings['production']):insufficient[part+'__'+name]+=1;continue
            trial=evaluated_fit(left,data,xy,time,sigma,keys,solution,center,config['reference_epoch'])
            values=dict(all_bands=(baseline['pm_icrs_masyr'],baseline['pm_error_masyr']),left_out=(trial['pm_icrs_masyr'],trial['pm_error_masyr']))
            deleted[part+'__'+name].append(values);leave[name]=trial
        private.append(dict(qso_local_index=row['qso_local_index'],magnitude=row['magnitude'],leave=leave))
    opts=dict(min_publish_count=5,tail_masyr=5,bootstrap_draws=400,bootstrap_absolute_zero=True,seed=9137)
    result=dict(deletions={k:comparison(v,opts,(('all_bands','left_out'),)) for k,v in deleted.items()},
        deleted_time_information_insufficient=dict(insufficient),
        residual_object_medians={k:dict(n=len(v),median_mas=np.median(v,axis=0),nmad_mas=1.4826*np.median(abs(np.asarray(v)-np.median(v,axis=0)),axis=0)) for k,v in residuals.items()},
        sampling_by_band={k:dict(n=len(v),columns=['measurements','utc18_nights','utc_years','span_years'],p16_p50_p84=np.quantile(v,[.16,.5,.84],axis=0)) for k,v in sampling.items()},
        slurm_job=os.environ['SLURM_JOB_ID'],
        scope='Identical frozen QSO members: remove each actual NSC band or calendar year without reassociation. QSO diagnostics only; no zero/error coefficients trained. Conditional fit residual medians are not unbiased held positions. Time/band deletions are coupled by real sampling; insufficient-time cases kept in counts.')
    check_aggregate(result);save_json(work/'allband_diagnostics_summary.json',result)
    save_json(work/'private_deletions.json',private)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('work',type=Path);p.add_argument('pilot',type=Path);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use a Slurm compute node')
    a.work.mkdir(parents=True,exist_ok=True)
    bind(a.work,'pilot_receipt.json',verify_directory(a.pilot,('private_fixed_qso.json.gz','allband_qso_summary.json')))
    with run_stage(a.work,PRECISION,'ordinary_fixed_qso','allband-diagnostics-v14',inputs=('pilot_receipt.json',),outputs=('allband_diagnostics_summary.json','private_deletions.json')) as go:
        if go:run(**vars(a))


if __name__=='__main__':main()
