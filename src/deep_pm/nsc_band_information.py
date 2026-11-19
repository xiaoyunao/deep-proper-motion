"""Fixed-source band deletion and NSC/early-SDSS information diagnostics."""
import argparse
from collections import Counter
import gzip
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .__main__ import tomllib
from .allband_qso import evaluated_fit
from .continuous_reference import PRECISION,SPATIAL,bind,read_offsets,reference_frame,verify_directory
from .io import save_json
from .nsc_comparison import comparison
from .nsc_information import night_labels,split_members
from .pilot_production import quality_detections
from .precision_v03 import calibration_groups
from .provenance import run_stage
from .public_summary import check_aggregate
from .year_opportunities import utc_year


def run(work,trial):
    config=tomllib.loads(PRECISION.read_text());settings=tomllib.loads(SPATIAL.read_text());settings.update(bootstrap_draws=800,min_publish_count=5,bootstrap_absolute_zero=True)
    source=(trial/'source_receipt.json').resolve().parent;graph=(trial/'new_reference_receipt.json').resolve().parent/'private_c0.npz';center=reference_frame(graph);solution=read_offsets(graph)
    data,xy,time,sigma,keep=quality_detections(source,center,config,settings['production']);lookup=pd.Index(data.detection_id);nsc=data.survey.eq('NSC').to_numpy();keys=calibration_groups(data);night=night_labels(data);year=utc_year(data.observation_time.to_numpy())
    with gzip.open(trial/'private_fixed_qso.json.gz','rt') as stream:frozen=json.load(stream)
    rows=[]
    for row in frozen:
        if not {'quality_r','all_nsc_w0'}<=row['members'].keys():continue
        base=lookup.get_indexer(row['members']['quality_r']);full=lookup.get_indexer(row['members']['all_nsc_w0'])
        if (np.r_[base,full]<0).any():raise ValueError('Missing frozen member')
        split_members(base,full,data,night)
        values={};late=full[nsc[full]];early=full[~nsc[full]];first=early[np.argmin(time[early])]
        choices=dict(all_nsc_sdss_all=full,nsc_r_only=base[nsc[base]],nsc_all_only=late,all_nsc_sdss_first=np.r_[late,first])
        bands=sorted(set(data.band.iloc[late])-{'r'})
        for band in bands:choices['leave_'+str(band)]=full[~(nsc[full]&data.band.iloc[full].eq(band).to_numpy())]
        for name,index in choices.items():
            if len(index)<2 or np.ptp(time[index])<.6:continue
            fit=evaluated_fit(index,data,xy,time,sigma,keys,solution,center,config['reference_epoch']);values[name]=(fit['pm_icrs_masyr'],fit['pm_error_masyr'])
        np.testing.assert_allclose(values['all_nsc_sdss_all'],row['values']['all_nsc_w0'],rtol=1e-9,atol=1e-8)
        original=set(night[base[nsc[base]]]);opportunity={}
        for band in bands:
            selected=late[data.band.iloc[late].eq(band).to_numpy()];new=~np.isin(night[selected],list(original))
            other=late[~data.band.iloc[late].eq(band).to_numpy()];other_nights=set(night[other]);shared=set(night[selected])&other_nights
            separation=[np.min(abs(time[i]-time[other[night[other]==night[i]]]))*365.25*86400 for i in selected if night[i] in shared]
            opportunity[band]=dict(measurements=len(selected),new_nights=len(set(night[selected[new]])),original_r_nights=len(set(night[selected[~new]])),
                years=sorted(set(map(int,year[selected]))),span_years=float(np.ptp(time[selected])),unique_nights_relative_other_bands=len(set(night[selected])-other_nights),
                shared_nights_with_other_bands=len(shared),same_night_nearest_other_band_seconds=float(np.median(separation)) if separation else None)
        rows.append(dict(magnitude=row['magnitude'],sdss_runs=row['sdss_runs'],values=values,opportunity=opportunity))
    cohorts={}
    for low in (20,21,22):
        part=[r for r in rows if low<=r['magnitude']<low+1];cohorts['r'+str(low)]={}
        required=('nsc_r_only','nsc_all_only','all_nsc_sdss_first','all_nsc_sdss_all')
        common=[{k:r['values'][k] for k in required} for r in part if set(required)<=r['values'].keys()]
        cohorts['r'+str(low)]['nsc_sdss_time_decomposition']=comparison(common,settings,(('nsc_all_only','nsc_r_only'),('all_nsc_sdss_first','nsc_all_only'),('all_nsc_sdss_all','all_nsc_sdss_first')))
        for band in sorted({b for r in part for b in r['opportunity']}):
            selected=[r for r in part if 'leave_'+band in r['values']];names=('all_nsc_sdss_all','leave_'+band)
            values=[{k:r['values'][k] for k in names} for r in selected]
            condition=[r['opportunity'][band] for r in selected]
            cohorts['r'+str(low)]['band_'+band]=dict(same_band_covered_objects=comparison(values,settings,((names[0],names[1]),)),
                contributed_time_p16_p50_p84=np.quantile([[r['measurements'],r['new_nights'],r['original_r_nights'],r['span_years']] for r in condition],[.16,.5,.84],axis=0) if condition else None,
                time_columns=['measurements','new_nights','original_r_nights','span_years'],source_support_by_year=dict(Counter(str(y) for r in condition for y in r['years'])))
            cohorts['r'+str(low)]['band_'+band]['relative_to_all_other_bands']=dict(
                unique_nights_p16_p50_p84=np.quantile([r['unique_nights_relative_other_bands'] for r in condition],[.16,.5,.84]) if condition else None,
                shared_nights_p16_p50_p84=np.quantile([r['shared_nights_with_other_bands'] for r in condition],[.16,.5,.84]) if condition else None,
                nearest_same_night_other_band_seconds_p16_p50_p84=np.quantile([r['same_night_nearest_other_band_seconds'] for r in condition if r['same_night_nearest_other_band_seconds'] is not None],[.16,.5,.84]) if any(r['same_night_nearest_other_band_seconds'] is not None for r in condition) else None)
    result=dict(cohorts=cohorts,fixed_sources=len(rows),slurm_job=os.environ['SLURM_JOB_ID'],scope='Identical frozen QSO/r/SDSS members and reference graph. Band deletions evaluated only on the same covered source subset, never comparing different successes. Every available NSC band diagnosed without a name whitelist. NSC-only/first-SDSS/all-SDSS use exactly the same NSC measurements. No formal-error postselection; these are conditional known-position controls, not blind purity or causal atmospheric estimates.')
    check_aggregate(result);save_json(work/'nsc_band_information_summary.json',result)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('work',type=Path);p.add_argument('trial',type=Path);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use a Slurm compute node')
    a.work.mkdir(parents=True,exist_ok=True);bind(a.work,'trial_receipt.json',verify_directory(a.trial,('private_fixed_qso.json.gz',)))
    with run_stage(a.work,PRECISION,'fixed_bands','nsc-band-information-v15',inputs=('trial_receipt.json',),outputs=('nsc_band_information_summary.json',)) as go:
        if go:run(a.work,a.trial)


if __name__=='__main__':main()
