"""Pool synchronized C0 reference errors and real spatial/condition diagnostics."""
import argparse
from collections import defaultdict
import gzip
import json
import os
from pathlib import Path

import healpy as hp
import numpy as np
import pandas as pd

from .astrometry import project
from .condition_residuals import aggregate_positions,spatial_mean
from .continuous_reference import reference_frame,verify_directory
from .continuous_summary import estimate,deletes
from .io import digest,save_json
from .public_summary import check_aggregate
from .standard_motion import icrs_motion
from .validation import ratio


def split_rows(rows):
    # Metadata-only identity split; never partition on fitted motion/residual.
    from hashlib import sha256
    train=[];test=[]
    for row in rows:
        identity=f"{row['core']}:{row['qso_index']}:9137"
        (train if int(sha256(identity.encode()).hexdigest()[:8],16)%2 else test).append(row)
    return train,test


def error_calibration(rows):
    train,test=split_rows(rows)
    if len(train)<50 or len(test)<50:return dict(training_count=len(train),test_count=len(test),status='insufficient_support')
    pm=np.array([r['motion'] for r in train]);err=np.array([r['error'] for r in train])
    z=pm/err
    scale=1.4826*np.median(abs(z-np.median(z,axis=0)),axis=0)
    pm=np.array([r['motion'] for r in test]);err=np.array([r['error'] for r in test])
    return dict(training_count=len(train),test_count=len(test),scale=scale,
                test_raw_within_1sigma=[ratio(int(n),len(test)) for n in (abs(pm)<=err).sum(axis=0)],
                test_scaled_within_1sigma=[ratio(int(n),len(test)) for n in (abs(pm)<=err*scale).sum(axis=0)],
                scope='Held QSO uncertainty diagnostic only; no motion correction or adopted error model. Source split can share physical calibration groups.')


def describe(rows):
    control=[dict(r,motion={'C0':r['motion']}) for r in rows]
    result=dict(spatial_zero=spatial_mean(rows) if len(rows)>=5 else dict(n=len(rows)),physical_group_sensitivity=estimate(control,('C0',),9137),
                source_error_calibration=error_calibration(rows),axis_zero_target_masyr=.5)
    # Deliberately do not infer certification from a linearized bootstrap alone.
    result['spatial_certification']='not_certified_pending_independent_group_and_spatial_systematic_tests'
    return result


def pool(frozen,conditions,spatial,output):
    motions=[];positions=[];receipts={};removed=[]
    center=reference_frame(frozen/'shared/private_c0.npz')
    if not np.allclose(center,[350.,0.],rtol=0,atol=1e-12):raise ValueError('Only the frozen C region is supported')
    for line in (frozen/'run-plan.tsv').read_text().splitlines():
        name=line.split()[0]
        work=spatial/name
        receipt=verify_directory(work,('private_motion_covariance.json.gz','private_position_residuals.json.gz','condition_residual_summary.json'))
        metadata=json.loads((work/'condition_residual_summary.json').read_text())
        calibration_sha=metadata['receipt']['shared_solution_sha256']
        reference_sha=digest(frozen/'shared/private_c0.npz') if not receipts else next(iter(receipts.values()))['reference_sha256']
        if calibration_sha!=reference_sha:raise ValueError('Reference draw synchronization changed')
        verify_directory(conditions/name,('private_conditions.json','conditions_summary.json'))
        timing={r['qso_index']:r for r in json.loads((conditions/name/'private_conditions.json').read_text())}
        with gzip.open(work/'private_motion_covariance.json.gz','rt') as stream:current=json.load(stream)
        for row in current:
            old=timing[row['qso_index']]
            np.testing.assert_allclose(row['motion'],old['motion']['all_SDSS'],rtol=0,atol=1e-7)
            pos=project([row['ra']],[row['dec']],center)[0]
            jac=icrs_motion(pos,np.zeros(2),center)['state_jacobian'][np.ix_([1,3],[1,3])]
            independent=jac@row['independent_pm_covariance']@jac.T
            common=np.asarray(row['reference_pm_draws'])@jac.T
            row.update(core=name,motion=jac@row['motion'],
                independent_pm_covariance=independent,reference_pm_draws=common,
                error=np.sqrt(np.diag(independent+np.cov(common,rowvar=False))),
                exposures=old['exposures'],nsc_epoch_days=old['nsc_epoch_days'],
                sdss_runs=old['sdss_runs'],nsc_span_years=old['nsc_span_years'])
            motions.append(row)
        with gzip.open(work/'private_position_residuals.json.gz','rt') as stream:
            positions.extend(dict(row,qso_index=f"{name}:{row['qso_index']}") for row in json.load(stream))
        removed.extend(json.loads((frozen/'outputs'/name/'private_group_deletions.json').read_text()))
        receipts[name]=dict(summary_sha256=digest(work/'condition_residual_summary.json'),receipt_sha256=digest(receipt),reference_sha256=reference_sha)
    groups=defaultdict(list)
    for row in motions:
        band='ordinary' if abs(row['dec'])>=3 else 'repeat' if abs(row['dec'])<=1 else 'transition'
        for label in ('all',band,'one_sdss_run' if row['sdss_runs']==1 else 'multiple_sdss_runs',
                      'nsc_span_lt_2yr' if row['nsc_span_years']<2 else 'nsc_span_ge_2yr'):
            groups[f"{label}__r{int(row['magnitude'])}_{int(row['magnitude'])+1}"].append(row)
    pixels=hp.ang2pix(16,[r['ra'] for r in motions],[r['dec'] for r in motions],lonlat=True)
    sky={str(k):describe([r for r,pixel in zip(motions,pixels) if pixel==k]) for k in sorted(set(pixels))}
    for key,value in sky.items():
        value['by_r']={f'r{low}_{low+1}':describe([r for r,pixel in zip(motions,pixels) if pixel==int(key) and low<=r['magnitude']<low+1]) for low in (21,22)}
    tagged=[]
    for row in positions:
        labels=[row['survey']+':'+str(row['instrument'])+':'+row['band'],f"r{int(row['r'])}_{int(row['r'])+1}",f"year_{row['year']}"]
        if row['color'] is not None:labels.append('color_blue' if row['color']<.5 else 'color_mid' if row['color']<1.2 else 'color_red')
        for key,cuts in [('fwhm',(1.,1.5)),('airmass',(1.3,1.6))]:
            if row[key] is not None:labels.append(key+('_low' if row[key]<cuts[0] else '_mid' if row[key]<cuts[1] else '_high'))
        tagged.extend(dict(row,diagnostic_group=label) for label in labels)
    public=dict(spatial_zero_minimum_qso_count=5,completed_cores=len(receipts),qso_count=len(motions),position_count=len(positions),receipts=receipts,
        cohorts={key:describe(rows) for key,rows in groups.items()},spatial_nside16=sky,
        position_residual_strata=aggregate_positions(pd.DataFrame(tagged)),
        fixed_reference_leave_group_out=deletes(removed),
        motion_scope='ICRS PM block rotation at fixed QSO positions; synchronized reference draw indices preserved throughout frozen C graph. No new correction applied.',
        limitation='Fixed-position PM Jacobian omits the tiny velocity-position derivative for stationary QSO; no native RA-Dec measurement covariance or empirical shared systematic certification. Legacy NSC night labels retain the original TT boundary.',
        slurm_job=os.environ['SLURM_JOB_ID'])
    check_aggregate(public);save_json(output,public)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('frozen','conditions','spatial','output'):p.add_argument(key,type=Path)
    a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use Slurm')
    pool(a.frozen,a.conditions,a.spatial,a.output)


if __name__=='__main__':main()
