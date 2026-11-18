"""Trace median-like extra-QSO cases; publish figures, never source tables."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .__main__ import tomllib
from .astrometry import project
from .condition_error_trial import families
from .continuous_reference import PRECISION,SPATIAL,read_offsets,reference_frame,verify_directory
from .dark_precision import query_unique
from .exposure_catalogues import coordinates,inside
from .io import digest,save_json
from .nsc_comparison import qso_matches
from .pilot_production import quality_detections
from .precision_v03 import calibration_groups
from .public_summary import check_aggregate


def render(work,batch,extra,members):
    plt.rcParams.update({'font.family':'serif','font.size':10,'savefig.dpi':180})
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    receipt={}
    for column,(core,box,tag,component,title) in enumerate([('b_16_00','b','22_to_23',0,'B: extra r=22–23'),('north_one_00_00','north_one','21_to_22',1,'North one: extra r=21–22')]):
        member=json.loads((members/(core+'_member_summary.json')).read_text())
        if any(member['member_counts'][k] for k in ('wrong_known_members','unknown_members','bad_quality_members')):raise ValueError('Need case-level member validation before selecting examples')
        trial=extra/'errors'/core
        verify_directory(trial,('private_extra_training_sources.json','error_trial_summary.json'))
        records=json.loads((trial/'private_extra_training_sources.json').read_text())
        summary=json.loads((trial/'error_trial_summary.json').read_text())
        spec=tomllib.loads((batch/f'batch-{box}.toml').read_text())['fields'][core]
        shared=batch/'shared'/box/'private_c0.npz';center=reference_frame(shared);offsets=read_offsets(shared)
        source=batch/'inputs'/core
        config=tomllib.loads(PRECISION.read_text());settings=tomllib.loads(SPATIAL.read_text())
        data,xy,time,sigma,keep=quality_detections(source,center,config,settings['production'])
        keys=calibration_groups(data);nsc=data.survey.eq('NSC').to_numpy()
        qso=pd.read_csv(source/'spectra_union.csv');qso=qso.loc[inside(qso.ra,qso.dec,spec)].reset_index(drop=True)
        positions=np.array([r['position_deg'] for r in records]);radius=lambda mas:2*np.sin(np.deg2rad(mas/3600000)/2)
        good,target=query_unique(cKDTree(coordinates(qso.ra.to_numpy(),qso.dec.to_numpy())),coordinates(*positions.T),radius(500),radius(300));counts=Counter(target[good])
        eligible=[i for i,r in enumerate(records) if good[i] and counts[target[i]]==1 and r['r_bin']==tag]
        if len(eligible)<5:raise ValueError('Insufficient cohort for representative selection')
        median=np.median([records[i]['pm_icrs_masyr'][component] for i in eligible])
        chosen=min(eligible,key=lambda i:abs(records[i]['pm_icrs_masyr'][component]-median));row=records[chosen]
        index=pd.Series(np.arange(len(data)),index=data.detection_id).loc[row['detection_ids']].to_numpy()
        year=summary['annual_same_errors_multiband']['year']
        from astropy.time import Time
        edges=Time([f'{year}-01-01',f'{year+1}-01-01'],scale='utc').tt.jyear
        _,held_truth=qso_matches(data,xy,keep&nsc&(time>=edges[0])&(time<edges[1]),project(qso.ra,qso.dec,center))
        held=held_truth.get(int(target[chosen]),np.array([],int));held=held[np.isin(keys[held],list(offsets))]
        if set(held)&set(index):raise ValueError('Held stellar positions entered training')
        point=project([row['position_deg'][0]],[row['position_deg'][1]],center)[0]
        jac=np.asarray(row['state_jacobian'])[np.ix_([1,3],[1,3])];velocity=np.linalg.solve(jac,row['pm_icrs_masyr'])
        raw=data[['sigma_east_mas','sigma_north_mas']].to_numpy(float);errors=np.hypot(raw,50.)
        family=families(data)
        for label,model in summary['error_models'].items():
            if model['enabled_for_trial_only']:
                params=np.asarray(model['parameters_scale_floor']);use=family==label
                errors[use]=np.sqrt((raw[use]*params[:,0])**2+params[:,1]**2)
        def values(use):
            pos=xy[use]-np.array([offsets[keys[i]][0] for i in use])-point
            covariance=np.array([np.diag(errors[i]**2)+np.cov(offsets[keys[i]][1],rowvar=False) for i in use])
            return pos@jac.T,np.sqrt(np.einsum('ab,nbc,ac->na',jac,covariance,jac))
        observed,err=values(index);held_value,held_err=values(held) if len(held) else (None,None)
        bands=np.where(nsc[index],data.band.iloc[index].to_numpy(),'SDSS r')
        colors={'SDSS r':'#555555','r':'#24689b','g':'#369b62','i':'#ce843d','z':'#884da7'}
        grid=np.linspace(min(time[index])-1,max(time[index])+1,200)
        for axis in (0,1):
            ax=axes[axis,column]
            for band in sorted(set(bands)):
                use=bands==band
                ax.errorbar(time[index[use]],observed[use,axis],yerr=err[use,axis],fmt='s' if band=='SDSS r' else 'o',ms=4,color=colors[band],alpha=.8,label=band if band=='SDSS r' else 'NSC '+band)
            ax.plot(grid,(grid-2015.5)*(jac@velocity)[axis],c='k',lw=1,label='Saved free-motion fit')
            stationary=np.average(observed[:,axis],weights=1/err[:,axis]**2)
            ax.axhline(stationary,c='.5',ls=':',label='Stationary control (evaluation only)')
            if len(held):ax.errorbar(time[held],held_value[:,axis],yerr=held_err[:,axis],fmt='*',mfc='none',mec='#b33233',ecolor='#b33233',ms=10,label=f'Held r ({year})')
            ax.set(ylabel=('East','North')[axis]+' displacement [mas]',xlabel='Observation Julian year (TT)',title=title if axis==0 else None)
            ax.grid(alpha=.2);ax.tick_params(direction='in',top=True,right=True)
        axes[0,column].legend(fontsize=7,ncol=2,frameon=False)
        receipt[box]=dict(cohort_count=len(eligible),selection_component=('east','north')[component],held_year=year,held_positions=len(held),input_summary_sha256=digest(trial/'error_trial_summary.json'))
    fig.suptitle('Median-like extra-QSO examples: band/time sampling and the held-year check',fontsize=11)
    fig.savefig(work/'extra_case_summary.png');plt.close(fig)
    public=dict(cohort_metadata=receipt,selection='Closest fitted component to the cohort median, not an extreme. No source identifiers or absolute sky coordinates exported.',scope='Marginal position bars include trial noise and C0 reference terms; measurements remain correlated. Individual nonzero motions need not be significant.',slurm_job=os.environ['SLURM_JOB_ID'])
    check_aggregate(public);save_json(work/'extra_case_summary.json',public)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('work','batch','extra','members'):p.add_argument(key,type=Path)
    a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use Slurm')
    a.work.mkdir(parents=True,exist_ok=True);render(a.work,a.batch,a.extra,a.members)


if __name__=='__main__':main()
