"""Independent post-fit spectral membership/quality audit of frozen blind sources."""
import argparse
from collections import Counter,defaultdict
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .__main__ import tomllib
from .allband_qso import physical_candidates
from .astrometry import project,unproject
from .continuous_reference import PRECISION,SPATIAL,bind,reference_frame,verify_directory
from .dark_precision import query_unique
from .exposure_catalogues import coordinates
from .io import save_json
from .nsc_comparison import qso_matches,comparison
from .pilot_production import quality_detections
from .provenance import run_stage
from .public_summary import check_aggregate
from .validation import ratio
from .year_opportunities import utc_year


def audit(work,trial):
    summary=json.loads((trial/'allband_blind_year_summary.json').read_text())
    if summary['status']!='completed':save_json(work/'allband_member_audit_summary.json',dict(status=summary['status']));return
    source=(trial/'source_receipt.json').resolve().parent;graph=(trial/'reference_receipt.json').resolve().parent/'private_c0.npz'
    config=tomllib.loads(PRECISION.read_text());settings=tomllib.loads(SPATIAL.read_text());settings.update(bootstrap_draws=800,bootstrap_absolute_zero=True)
    center=reference_frame(graph);data,xy,time,sigma,keep=quality_detections(source,center,config,settings['production']);nsc=data.survey.eq('NSC').to_numpy()
    year=utc_year(data.observation_time.to_numpy());training_mask=(keep|(nsc&physical_candidates(data)))&~(nsc&(year==summary['year']))
    qso=pd.read_csv(source/'spectra_union.csv');qso=qso.loc[np.isfinite(qso[['ra','dec']]).all(axis=1)].reset_index(drop=True)
    label,_=qso_matches(data,xy,training_mask,project(qso.ra,qso.dec,center))
    records=json.loads((trial/'private_training.json').read_text());lookup=pd.Index(data.detection_id)
    good=np.zeros(len(records),bool);nearest=np.full(len(records),-1,int)
    if records and len(qso):
        sky=unproject([r['position'] for r in records],center);radius=lambda mas:2*np.sin(np.deg2rad(mas/3600000)/2)
        good,nearest=query_unique(cKDTree(coordinates(qso.ra.to_numpy(),qso.dec.to_numpy())),coordinates(*sky.T),radius(500),radius(300))
    claims=Counter(nearest[good]);selected=[];source_counts=Counter(r['kind']+'__r'+str(int(r['magnitude'])) for r in records)
    quality=physical_candidates(data);trees={};visits=data.exposure_id.to_numpy();by_visit={name:part.index.to_numpy() for name,part in data.loc[training_mask].groupby('exposure_id',sort=False)}
    for j,row in enumerate(records):
        if not good[j] or claims[nearest[j]]!=1:continue
        ix=lookup.get_indexer(row['detection_ids'])
        if (ix<0).any() or np.any(nsc[ix]&(year[ix]==summary['year'])):raise ValueError('Saved source changed or held-year leakage')
        known=label[ix];wrong=int(((known>=0)&(known!=nearest[j])).sum());unknown=int((known<0).sum())
        consistent=wrong==0 and unknown==0 and nsc[ix].any() and (~nsc[ix]).any();nearby=0
        for k in ix:
            name=visits[k]
            if name not in trees:trees[name]=cKDTree(xy[by_visit[name]])
            distances,_=trees[name].query(xy[k],k=2)
            nearby+=bool(distances[1]<1000)
        flags=pd.to_numeric(data.quality_flags.iloc[ix],errors='coerce').fillna(0).to_numpy(np.int64)
        selected.append(dict(kind=row['kind'],magnitude=row['magnitude'],values=(row['pm_icrs_masyr'],row['pm_error_masyr']),
            members=len(ix),known_competing_members=wrong,unknown_members=unknown,all_members_consistent=consistent,
            nonphysical_members=int((~quality[ix]).sum()),neighbor_within_1arcsec_members=nearby,
            nsc_deblended_members=int(np.count_nonzero(nsc[ix]&((flags&2)!=0)))))
    cohorts={}
    for key,total in sorted(source_counts.items()):
        kind,mag=key.split('__');low=int(mag[1:]);rows=[r for r in selected if r['kind']==kind and low<=r['magnitude']<low+1]
        n=len(rows);members=sum(r['members'] for r in rows);wrong=sum(r['known_competing_members'] for r in rows);unknown=sum(r['unknown_members'] for r in rows)
        cohorts[key]=dict(frozen_training_sources=total,unique_spectral_sources=n,non_spectral_identity_unverified_sources=total-n,
            consistent_sources=ratio(sum(r['all_members_consistent'] for r in rows),n),
            known_competing_member_fraction=ratio(wrong,members),unknown_member_fraction=ratio(unknown,members),
            source_with_neighbor_or_deblend=ratio(sum(r['neighbor_within_1arcsec_members']>0 or r['nsc_deblended_members']>0 for r in rows),n),
            source_with_nonphysical_member=ratio(sum(r['nonphysical_members']>0 for r in rows),n),
            all_spectral_zero=comparison([dict(multi=r['values']) for r in rows],settings),
            consistent_spectral_zero=comparison([dict(multi=r['values']) for r in rows if r['all_members_consistent']],settings))
    result=dict(status='completed',year=summary['year'],frozen_training_count=len(records),unique_spectral_sources=len(selected),by_class=cohorts,
        slurm_job=os.environ['SLURM_JOB_ID'],
        scope='Post-fit stationary spectral positions identify source and individual training members; PM similarity never labels identity. Unknown members and nonspectral sources are not declared correct. Neighbor/deblend indicates suspicion, not image-level blend truth. Zero known errors in finite samples retains Wilson intervals. No source selection or fitting changed.',uncertainty_scope='Spectral-zero intervals resample objects only. Shared physical groups require separate uncertainty/zero certification; their common error cannot be averaged down by source count.',adoption='audit_only_not_accepted_dark_star_counts')
    check_aggregate(result);save_json(work/'allband_member_audit_summary.json',result)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('work',type=Path);p.add_argument('trial',type=Path);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use a Slurm compute node')
    a.work.mkdir(parents=True,exist_ok=True);bind(a.work,'source_trial_receipt.json',verify_directory(a.trial,('private_training.json','allband_blind_year_summary.json')))
    with run_stage(a.work,PRECISION,'allband_blind_members','allband-members-v14',inputs=('source_trial_receipt.json',),outputs=('allband_member_audit_summary.json',)) as go:
        if go:audit(**vars(a))


if __name__=='__main__':main()
