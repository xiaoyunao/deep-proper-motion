"""Leave-group QSO position residuals and correlated spatial motion diagnostics."""
import argparse
from collections import defaultdict
import gzip
import json
import os
from pathlib import Path
import subprocess

import healpy as hp
import numpy as np
import pandas as pd

from .__main__ import tomllib
from .astrometry import project
from .continuous_reference import FILES,PRECISION,SPATIAL,bind,read_offsets,reference_frame,verify_directory
from .exposure_catalogues import inside
from .io import digest,json_default,save_json
from .nsc_comparison import qso_matches
from .pilot_production import quality_detections,valid_epochs
from .precision_v03 import calibration_groups,fit_with_calibration
from .provenance import run_stage
from .public_summary import check_aggregate
from .standard_motion import calibrated_covariance


def spatial_mean(records):
    """Mean zero with independent and synchronous reference contributions separate."""
    if not records:
        return dict(n=0)
    pm=np.array([r['motion'] for r in records])
    independent=np.array([r['independent_pm_covariance'] for r in records])
    draws=np.array([r['reference_pm_draws'] for r in records])
    precision=1/np.diagonal(independent,axis1=1,axis2=2)
    weights=precision/precision.sum(axis=0)
    zero=(pm*weights).sum(axis=0)
    common_draw=(draws*weights[:,None,:]).sum(axis=0)
    common=np.cov(common_draw,rowvar=False)
    ind=np.einsum('ia,iab,ib->ab',weights,independent,weights)
    error=np.sqrt(np.diag(ind+common))
    return dict(n=len(records),weighted_mean_masyr=zero,independent_mean_covariance=ind,
                reference_mean_covariance=common,model_mean95_masyr=np.stack([zero-1.96*error,zero+1.96*error],axis=1),
                median_masyr=np.median(pm,axis=0),nmad_masyr=1.4826*np.median(abs(pm-np.median(pm,axis=0)),axis=0),
                qualification='Linearized model interval; unmodeled spatial systematics and physical-block evidence still required.')


def aggregate_positions(frame):
    output={}
    for name,group in frame.groupby('diagnostic_group',sort=True):
        if group.qso_index.nunique()<5:
            output[name]=dict(position_count=len(group),qso_count=group.qso_index.nunique())
            continue
        value=group[['residual_e','residual_n']].to_numpy()
        sigma=group[['sigma_e','sigma_n']].to_numpy()
        median=np.median(value,axis=0)
        output[name]=dict(position_count=len(group),qso_count=group.qso_index.nunique(),median_mas=median,
            nmad_mas=1.4826*np.median(abs(value-median),axis=0),
            radial_p68_mas=np.quantile(np.linalg.norm(value,axis=1),.68),
            normalized_nmad=1.4826*np.median(abs(value/sigma-np.median(value/sigma,axis=0)),axis=0),
            within_1sigma_count=(abs(value)<=sigma).sum(axis=0))
    return output


def run(work,source,shared,spec):
    config=tomllib.loads(PRECISION.read_text())
    settings=tomllib.loads(SPATIAL.read_text())
    center=reference_frame(shared/'private_c0.npz')
    data,xy,time,sigma,keep=quality_detections(source,center,config,settings['production'])
    keys=calibration_groups(data)
    nsc=data.survey.eq('NSC').to_numpy()
    offsets,gauges=read_offsets(shared/'private_c0.npz',with_gauge=True)
    counts=pd.Series(keys[keep&inside(data.ra_deg,data.dec_deg,spec)]).map(gauges).value_counts()
    gauge=min(counts.index,key=lambda k:(-counts[k],k)) if len(counts) else None
    offsets={k:v for k,v in offsets.items() if gauges[k]==gauge}
    support=np.isin(keys,list(offsets))
    qso=pd.read_csv(source/'spectra_union.csv')
    qso=qso.loc[inside(qso.ra,qso.dec,spec)].reset_index(drop=True)
    points=project(qso.ra,qso.dec,center)
    _,rtruth=qso_matches(data,xy,keep,points)
    expanded=keep|(nsc&data.band.isin(['g','i','z']).to_numpy()&data.magnitude_error.between(0,.21714724095,inclusive='right').to_numpy())
    _,truth=qso_matches(data,xy,expanded,points)
    coord=xy-np.stack([offsets[k][0] if k in offsets else np.zeros(2) for k in keys])
    positions,motions=[],[]
    for source_index,ix in rtruth.items():
        ix=ix[support[ix]]
        if not 20<=qso.reference_r.iloc[source_index]<23 or not valid_epochs(ix,nsc,time,settings['production']):
            continue
        fit,errors=fit_with_calibration(xy[ix],time[ix],sigma[ix],config['reference_epoch'],[offsets[keys[j]] for j in ix])
        _,independent,_,influence=calibrated_covariance(fit,time[ix],sigma[ix],[offsets[keys[j]] for j in ix])
        samples=np.stack([offsets[keys[j]][1] for j in ix],axis=1)
        draws=-np.einsum('pna,dna->dp',influence,samples)[:,[1,3]]
        motions.append(dict(qso_index=source_index,magnitude=float(qso.reference_r.iloc[source_index]),
            ra=float(qso.ra.iloc[source_index]),dec=float(qso.dec.iloc[source_index]),motion=fit.coeff[1],error=errors,
            independent_pm_covariance=independent[np.ix_([1,3],[1,3])],reference_pm_draws=draws,
            groups=sorted(set(keys[ix])),ring=int(hp.ang2pix(128,qso.ra.iloc[source_index],qso.dec.iloc[source_index],lonlat=True))))
        color=data.color.iloc[ix] if 'color' in data else pd.Series(dtype=float)
        color=float(color.median()) if color.notna().any() else None
        if color is not None and (not np.isfinite(color) or not -3<=color<=6):
            color=None
        all_index=truth[source_index]
        for i in all_index[support[all_index]]:
            train=ix[keys[ix]!=keys[i]]
            if len(train)<2:
                continue
            weight=1/sigma[train]**2
            weight/=weight.sum(axis=0)
            prediction=(coord[train]*weight).sum(axis=0)
            reference_draw=offsets[keys[i]][1]-np.einsum('na,dna->da',weight,
                np.stack([offsets[keys[j]][1] for j in train],axis=1))
            variance=sigma[i]**2+np.sum(weight**2*sigma[train]**2,axis=0)+np.var(reference_draw,axis=0,ddof=1)
            value=coord[i]-prediction
            row=dict(qso_index=source_index,group=keys[i],band=str(data.band.iloc[i]),survey=str(data.survey.iloc[i]),
                r=float(qso.reference_r.iloc[source_index]),color=color,year=int(np.floor(time[i])),
                residual_e=float(value[0]),residual_n=float(value[1]),sigma_e=float(np.sqrt(variance[0])),sigma_n=float(np.sqrt(variance[1])))
            for key in ('instrument','fwhm','airmass','x','y','field_id','detector_id'):
                v=data[key].iloc[i] if key in data else None
                row[key]=v if pd.notna(v) else None
                if key!='instrument' and row[key] is not None:
                    bounds={'fwhm':(0.,10.),'airmass':(1.,5.),'x':(0.,50000.),'y':(0.,50000.),
                            'field_id':(0.,1000000.),'detector_id':(0.,10000.)}[key]
                    if not np.isfinite(v) or not bounds[0]<=v<=bounds[1]:
                        row[key]=None
            positions.append(row)
    with gzip.open(work/'private_position_residuals.json.gz','wt') as f:
        json.dump(positions,f,default=json_default,allow_nan=False)
    with gzip.open(work/'private_motion_covariance.json.gz','wt') as f:
        json.dump(motions,f,default=json_default,allow_nan=False)
    strata=defaultdict(list)
    for p in positions:
        tags=[p['survey']+':'+str(p['instrument'])+':'+p['band'],f"r{int(p['r'])}_{int(p['r'])+1}",
              f"year_{p['year']}","physical_group:"+p['group']]
        if p['detector_id'] is not None:
            tags.append(p['survey']+':'+str(p['instrument'])+':detector_'+str(p['detector_id']))
        if p['field_id'] is not None:
            tags.append('field:'+p['group']+':'+str(p['field_id']))
        if p['color'] is not None:
            tags.append('color_blue' if p['color']<.5 else 'color_mid' if p['color']<1.2 else 'color_red')
        for key,cuts in [('fwhm',(1.,1.5)),('airmass',(1.3,1.6))]:
            if p[key] is not None and np.isfinite(p[key]):
                tags.append(key+('_low' if p[key]<cuts[0] else '_mid' if p[key]<cuts[1] else '_high'))
        for tag in tags:
            strata[tag].append(dict(p,diagnostic_group=tag))
    diagnostics=aggregate_positions(pd.DataFrame([p for group in strata.values() for p in group])) if strata else {}
    pixels={str(k):spatial_mean([m for m in motions if m['ring']==k]) for k in sorted(set(m['ring'] for m in motions))}
    output=dict(qso_count=len(motions),position_count=len(positions),position_residual_strata=diagnostics,
                spatial_zero=spatial_mean(motions),nside128_spatial_zero=pixels,
                finite_metadata_counts={k:sum(p[k] is not None for p in positions) for k in ('color','fwhm','airmass','x','y','field_id')},
                residual_scope='Stationary spectral QSO; target group excluded; r-only reference positions, no mean correction trained or applied.')
    check_aggregate(output)
    save_json(work/'condition_residual_summary.json',output)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('work','source','shared','config'):
        p.add_argument(name,type=Path)
    p.add_argument('region')
    a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        p.error('Use a Slurm compute node')
    a.work.mkdir(parents=True,exist_ok=True)
    bind(a.work,'source_receipt.json',verify_directory(a.source,FILES))
    bind(a.work,'shared_receipt.json',verify_directory(a.shared,('private_c0.npz',)))
    with run_stage(a.work,a.config,a.region,'conditions-residual-v13',inputs=('source_receipt.json','shared_receipt.json'),
        outputs=('private_position_residuals.json.gz','private_motion_covariance.json.gz','condition_residual_summary.json')) as go:
        if go:
            run(a.work,a.source,a.shared,tomllib.loads(a.config.read_text())['fields'][a.region])
            summary=json.loads((a.work/'condition_residual_summary.json').read_text())
            summary['receipt']=dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                slurm_job=os.environ['SLURM_JOB_ID'],configuration_sha256=digest(a.config),
                source_receipt_sha256=digest(a.work/'source_receipt.json'),
                shared_solution_sha256=digest(a.shared/'private_c0.npz'))
            save_json(a.work/'condition_residual_summary.json',summary)


if __name__=='__main__':
    main()
