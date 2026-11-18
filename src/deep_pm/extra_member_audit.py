"""Audit saved extra multiband sources with independent spectral member labels."""
import argparse
from collections import Counter,defaultdict
import json
import os
from pathlib import Path

from astropy.time import Time
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .__main__ import tomllib
from .astrometry import project
from .astrometry_audit import quality_flags
from .continuous_reference import FILES,PRECISION,SPATIAL,reference_frame,verify_directory
from .dark_precision import query_unique
from .exposure_catalogues import coordinates,inside
from .io import digest,save_json
from .nsc_comparison import qso_matches,comparison
from .pilot_production import quality_detections
from .public_summary import check_aggregate
from .validation import ratio


def audit(trial,source,shared,geometry,region,output):
    verify_directory(trial,('error_trial_summary.json','private_extra_training_sources.json'))
    verify_directory(source,FILES)
    spec=tomllib.loads(geometry.read_text())['fields'][region]
    center=reference_frame(shared/'private_c0.npz')
    summary=json.loads((trial/'error_trial_summary.json').read_text())
    records=json.loads((trial/'private_extra_training_sources.json').read_text())
    if summary['annual_same_errors_multiband'].get('status')!='completed':
        save_json(output,dict(status='no_completed_year_validation'));return
    year=summary['annual_same_errors_multiband']['year']
    config=tomllib.loads(PRECISION.read_text());settings=tomllib.loads(SPATIAL.read_text())
    settings.update(bootstrap_draws=800,bootstrap_absolute_zero=True)
    data,xy,time,sigma,keep=quality_detections(source,center,config,settings['production'])
    nsc=data.survey.eq('NSC').to_numpy()
    expanded=keep|(nsc&data.band.isin(['g','i','z']).to_numpy()&data.magnitude_error.between(0,2.5/np.log(10)/settings['production']['min_snr'],inclusive='right').to_numpy())
    edges=Time([f'{year}-01-01',f'{year+1}-01-01'],scale='utc').tt.jyear
    train=expanded&~(nsc&(time>=edges[0])&(time<edges[1]))
    qso=pd.read_csv(source/'spectra_union.csv')
    qso=qso.loc[inside(qso.ra,qso.dec,spec)].reset_index(drop=True)
    labels,truth=qso_matches(data,xy,train,project(qso.ra,qso.dec,center))
    claims=Counter(i for indices in truth.values() for i in indices)
    ambiguous=[i for i,n in claims.items() if n>1]
    labels[ambiguous]=-1
    lookup=pd.Series(np.arange(len(data)),index=data.detection_id)
    clean=quality_flags(data)
    uncertain_flags=(~nsc)&data.quality_flags.abs().gt(2**53).to_numpy() if data.quality_flags.dtype.kind=="f" else np.zeros(len(data),bool)
    clean[uncertain_flags]=False
    if records:
        pos=np.asarray([r['position_deg'] for r in records]);radius=lambda mas:2*np.sin(np.deg2rad(mas/3600000)/2)
        good,target=query_unique(cKDTree(coordinates(qso.ra.to_numpy(),qso.dec.to_numpy())),coordinates(*pos.T),radius(500),radius(300))
        counts=Counter(target[good])
    else:good=np.array([],bool);target=np.array([],int);counts={}
    selected=[];detected=Counter();bands=defaultdict(Counter)
    for i,row in enumerate(records):
        if not good[i] or counts[target[i]]!=1:continue
        use=lookup.loc[row['detection_ids']].to_numpy()
        if not train[use].all():raise ValueError('Saved extra track contains held-year or noneligible measurement')
        known=labels[use]
        matched=known==target[i]
        wrong=int(((known>=0)&~matched).sum());unknown=int((known<0).sum())
        endpoints=bool(nsc[use[matched]].any() and (~nsc[use[matched]]).any())
        status='all_members_known_consistent' if wrong==0 and unknown==0 and endpoints else 'unknown_members_or_endpoints' if wrong==0 else 'known_competing_members'
        selected.append(dict(row,member_status=status,bad_quality=bool((~clean[use]).any())))
        detected.update(total_members=len(use),wrong_known_members=wrong,unknown_members=unknown,
                        bad_quality_members=int((~clean[use]).sum()))
        for j in use:
            key=str(data.survey.iloc[j])+':'+str(data.band.iloc[j])
            bands[key].update(total=1,bad_quality=int(not clean[j]))
    groups={}
    for tag in ('20_to_21','21_to_22','22_to_23'):
        rows=[r for r in selected if r['r_bin']==tag]
        outcomes={}
        for name,part in [('all',rows),('all_members_known_consistent',[r for r in rows if r['member_status']=='all_members_known_consistent']),('consistent_and_no_bad_flags',[r for r in rows if r['member_status']=='all_members_known_consistent' and not r['bad_quality']])]:
            outcomes[name]=comparison([dict(multi=(r['pm_icrs_masyr'],r['pm_error_masyr'])) for r in part],settings)
        groups[tag]=dict(count=len(rows),member_status_counts=dict(Counter(r['member_status'] for r in rows)),
                        with_bad_quality_flags=ratio(sum(r['bad_quality'] for r in rows),len(rows)),spectral_zero=outcomes)
    public=dict(source_summary_sha256=digest(trial/'error_trial_summary.json'),held_year=year,
        unique_spectral_sources=len(selected),ambiguous_validation_measurements=len(ambiguous),
        member_counts=dict(detected),floating_large_sdss_flag_count=int(uncertain_flags.sum()),member_quality_by_survey_band={k:dict(v) for k,v in bands.items()},cohorts=groups,
        scope='Post-fit member labels use only spectral positions and training detections. No fit or association rerun. Unknown labels are not assumed correct; clean subcohorts are audits, not a newly adopted selection.',
        flags_scope='Descriptive legacy quality_flags mask; unverified floating flags above 2^53 count as not clean (NSC saturation/truncation/etc; SDSS EDGE/SATURATED and children); original C0 and trial fits did not apply this veto.',
        slurm_job=os.environ['SLURM_JOB_ID'])
    check_aggregate(public);save_json(output,public)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('trial','source','shared','geometry'):p.add_argument(key,type=Path)
    p.add_argument('region');p.add_argument('output',type=Path);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use Slurm')
    audit(a.trial,a.source,a.shared,a.geometry,a.region,a.output)


if __name__=='__main__':main()
