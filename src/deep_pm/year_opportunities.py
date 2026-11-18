"""Separate geometric chip opportunities from usable held-year measurements."""
import argparse
import json
import os
from pathlib import Path

from astropy.time import Time
import numpy as np
import pandas as pd

from .__main__ import tomllib
from .access import Cache
from .continuous_reference import PRECISION,SPATIAL,read_offsets,reference_frame,verify_directory
from .io import digest,save_json
from .pilot_production import quality_detections
from .precision_v03 import calibration_groups,calibration_support
from .public_summary import check_aggregate
from .regional_preflight import r_chips
from .regional_production import r_chip_shapes


def utc_year(mjd):
    unique,inverse=np.unique(mjd,return_inverse=True)
    return np.array([int(s[:4]) for s in Time(unique,format='mjd',scale='tt').utc.isot])[inverse]


def audit(work,source,shared,geometry,region,output):
    summary=json.loads((work/'year_prediction_summary.json').read_text())
    if summary['status']!='completed':
        save_json(output,dict(status=summary['status']));return
    year=summary['year'];graph=shared/'private_c0.npz';center=reference_frame(graph)
    offsets,gauges=read_offsets(graph,with_gauge=True)
    offsets={k:v for k,v in offsets.items() if gauges[k]==summary['reference_gauge']}
    config=tomllib.loads(PRECISION.read_text());settings=tomllib.loads(SPATIAL.read_text())
    spec=tomllib.loads(geometry.read_text())['fields'][region]
    data,_,_,_,keep=quality_detections(source,center,config,settings['production'])
    nsc=data.survey.eq('NSC').to_numpy();held=nsc&(utc_year(data.observation_time.to_numpy())==year)
    support=calibration_support(calibration_groups(data),offsets)
    eligible=held&keep&support
    raw=pd.read_csv(source/'detections.csv',low_memory=False)
    raw=raw.loc[raw.survey.eq('NSC')&raw.band.eq('r')&np.isfinite(raw.observation_time)].copy()
    raw=raw.loc[utc_year(raw.observation_time.to_numpy())==year]
    local_center=tuple(float(np.mean(spec[k])) for k in ('ra','dec'))
    radius=float(np.hypot(np.ptp(spec['ra']),np.ptp(spec['dec']))/2+.06)
    chips=r_chips(Cache(work/'footprint_cache'),local_center,radius)
    keys='NSC:'+chips.exposure.astype(str)
    scopes={'all_archive_r_chips':np.ones(len(chips),bool),
        'supported_reference_groups':keys.isin(offsets).to_numpy(),
        'retrieved_eligible_exposures':keys.isin(set(data.exposure_id.loc[eligible])).to_numpy()}
    training=json.loads((work/'private_training.json').read_text())
    coverage={}
    for name,mask in scopes.items():
        covered=np.zeros(len(training),bool);visits=0
        for midpoint,polygons in r_chip_shapes(chips.loc[mask].reset_index(drop=True),center):
            if int(Time(midpoint,format='jyear',scale='tt').utc.isot[:4])!=year:continue
            visits+=1
            if not training:continue
            pred=np.array([r['position'] for r in training])+np.array([r['motion'] for r in training])*(midpoint-config['reference_epoch'])
            for polygon in polygons:covered |= polygon.contains_points(pred)
        coverage[name]=dict(held_visits=visits,by_r={tag:dict(train_sources=sum(r['r_bin']==tag for r in training),
            covered_sources=int(sum(ok and r['r_bin']==tag for r,ok in zip(training,covered))))
            for tag in ('20_to_21','21_to_22','22_to_23')})
    result=dict(year=year,raw_retrieved_r_count=len(raw),finite_native_error_r_count=int((held&data.band.eq('r').to_numpy()).sum()),
        snr_qualified_r_count=int((held&keep).sum()),reference_supported_r_count=int(eligible.sum()),
        coverage=coverage,usable_exposure_counts=sorted(map(int,data.loc[eligible].groupby('exposure_id').size())),
        input_receipt_sha256=digest(verify_directory(source,('detections.csv',))),
        year_summary_sha256=digest(work/'year_prediction_summary.json'),diagnostic_source_sha256=digest(Path(__file__)),
        scope='Metadata and fixed saved training predictions only. No fitting or new selection. Chip presence alone does not establish a retrievable, SNR/error-qualified, reference-supported detection.',
        slurm_job=os.environ['SLURM_JOB_ID'])
    check_aggregate(result);save_json(output,result)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('work','source','shared','geometry','region','output'):p.add_argument(name,type=str if name=='region' else Path)
    a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use a Slurm compute node')
    audit(**vars(a))


if __name__=='__main__':main()
