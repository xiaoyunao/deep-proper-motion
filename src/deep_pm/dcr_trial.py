"""Small fixed-reference geometric DCR experiment, not an adopted regional frame."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.time import Time

from .__main__ import tomllib
from .continuous_instruments import chip_receipts,instrument_by_exposure
from .continuous_reference import PRECISION,SPATIAL,bind,read_offsets,reference_frame,verify_directory,calibration_components
from .dcr_geometry import sdss_colors,combine_color,zenith_vectors,tangent_geometry,BLANCO
from .dcr_reference_fit import fit
from .io import save_json,digest
from .joint_reference import identity_split
from .pilot_production import quality_detections
from .precision_v03 import calibration_groups
from .provenance import run_stage
from .public_summary import check_aggregate


def prepare(work,pilot,reference,sas):
    source=(pilot/'source_receipt.json').resolve().parent;graph=(pilot/'new_reference_receipt.json').resolve().parent/'private_c0.npz';center=reference_frame(graph)
    config=tomllib.loads(PRECISION.read_text());settings=tomllib.loads(SPATIAL.read_text());data,xy,time,sigma,keep=quality_detections(source,center,config,settings['production']);keys=calibration_groups(data)
    base,gauges=read_offsets(graph,with_gauge=True)
    refs=pd.read_csv(reference/'reference.csv.gz',dtype={'reference':str,'is_holdout':bool});original_hash=identity_split(refs);index=pd.Index(data.detection_id).get_indexer(refs.detection_id)
    mask=(index>=0)&refs.group.isin(base).to_numpy();refs=refs.loc[mask].copy();index=index[mask]
    if not np.array_equal(refs.group.to_numpy(),keys[index]):raise ValueError('Original physical reference groups changed')
    vote=refs.loc[~refs.is_holdout,'group'].map(gauges).value_counts();gauge=vote.index[0];refs=refs.loc[refs.group.map(gauges)==gauge].copy();index=pd.Index(data.detection_id).get_indexer(refs.detection_id)
    # Same original sky detections, expressed in the actual common C0 basis.
    refs[['east','north']]=xy[index];refs[['sigma_e','sigma_n']]=sigma[index]
    phot,receipts=sdss_colors(data.loc[data.survey.eq('SDSS'),'detection_id'],sas);phot.to_csv(work/'private_color_photometry.csv',index=False);save_json(work/'private_photo_file_receipts.json',receipts)
    joined=refs.loc[refs.exposure_id.str.startswith('SDSS:'),['reference','detection_id']].merge(phot,on='detection_id',how='left',validate='many_to_one')
    reference_color={}
    for name,part in joined.groupby('reference',sort=False):reference_color[name]=combine_color(part)
    refs['color']=[reference_color.get(k,(np.nan,np.nan))[0] for k in refs.reference];refs['color_error']=[reference_color.get(k,(np.nan,np.nan))[1] for k in refs.reference]
    chips,complete=chip_receipts((source/'run_manifest.json').resolve());instruments=instrument_by_exposure(chips)
    ns=data.survey.eq('NSC');visits=data.loc[ns,['exposure_id','observation_time','band']].groupby('exposure_id',sort=True).agg(observation_time=('observation_time','median'),band=('band','first'))
    visits['instrument']=[instruments.get(k,'unknown') for k in visits.index];visits['role']=(visits.band=='g')&(visits.instrument=='c4d')
    vector=np.zeros((len(visits),3));active=visits.role.to_numpy();vector[active]=zenith_vectors(Time(visits.observation_time.to_numpy()[active],format='mjd',scale='tt').utc.mjd)
    for j in range(3):visits['zenith_'+str(j)]=vector[:,j]
    visits.to_csv(work/'private_group_geometry.csv');mapping={k:row for k,row in visits.iterrows()};h=np.zeros((len(refs),2));secz=np.full(len(refs),np.nan)
    role=np.array([bool(mapping[k].role) if k in mapping else False for k in refs.exposure_id]);selected=np.flatnonzero(role)
    if len(selected):
        vv=np.array([[mapping[k]['zenith_'+str(j)] for j in range(3)] for k in refs.exposure_id.iloc[selected]])
        h[selected],secz[selected]=tangent_geometry(refs.ra_deg.to_numpy()[selected],refs.dec_deg.to_numpy()[selected],vv,center)
    refs[['geometry_e','geometry_n']]=h;refs['geometric_airmass']=secz;refs['band']=data.band.iloc[index].to_numpy();refs['instrument']=[instruments.get(k,'SDSS' if k.startswith('SDSS:') else 'unknown') for k in refs.exposure_id]
    for k in ('airmass','fwhm','x','y','detector_id'):
        refs[k]=data[k].iloc[index].to_numpy() if k in data else np.nan
    refs['year']=np.floor(time[index]).astype(int);refs.to_csv(work/'private_reference_dcr.csv.gz',index=False)
    result=dict(original_reference_identity_split_sha256=original_hash,selected_reference_identity_split_sha256=identity_split(refs),reference_measurements=len(refs),training_references=refs.loc[~refs.is_holdout,'reference'].nunique(),held_references=refs.loc[refs.is_holdout,'reference'].nunique(),known_color_training_references=refs.loc[~refs.is_holdout&refs.color_error.notna(),'reference'].nunique(),physical_groups=refs.group.nunique(),reference_gauge=gauge,projection_center_deg=center,chip_receipts_complete=complete,physical_sdss_fields=len(receipts),slurm_job=os.environ['SLURM_JOB_ID'],
        scope='Exact original physical reference identities and split; retain original supported gauge and detections. Local D0/D1 solve is a controlled small-region experiment, not a replacement for N1/N3 global C0. Only existing SDSS PSF photometry/errors read; no additional SDSS positions. One scalar coefficient for DECam g with real zenith geometry; other instrument/bands unchanged except the jointly refit physical zeros.')
    check_aggregate(result);save_json(work/'dcr_preparation_summary.json',result)


def solve(work,workers):
    rows=pd.read_csv(work/'private_reference_dcr.csv.gz',dtype={'reference':str,'is_holdout':bool});config=tomllib.loads(PRECISION.read_text());cal=config['calibration'];train=rows.loc[~rows.is_holdout]
    pieces=list(calibration_components(train,cal['min_references']));part=max(pieces,key=lambda p:p.reference.nunique()).reset_index(drop=True)
    metadata=json.loads((work/'dcr_preparation_summary.json').read_text());center=metadata['projection_center_deg'];geometry=pd.read_csv(work/'private_group_geometry.csv').set_index('exposure_id')
    models={};coefficient={};support={};color_info={}
    for name,enabled in [('D0_local_C0',False),('D1_geometric_DCR',True)]:
        offset,beta,beta_draw,info=fit(part.reference,part[['east','north']].to_numpy(),part[['sigma_e','sigma_n']].to_numpy(),part.group.to_numpy(),part[['geometry_e','geometry_n']].to_numpy(),part.color.to_numpy(),part.color_error.to_numpy(),cal['bootstrap_draws'],config['seed'],workers,with_dcr=enabled)
        late=[v for k,v in offset.items() if k.startswith('NSC:')];anchor=np.median([v[0] for v in late],axis=0);datum=np.median([v[1]-v[1].mean(axis=0) for v in late],axis=0)
        kept={}
        for key,(point,samples,n) in offset.items():
            samples=samples-samples.mean(axis=0)-datum
            if np.std(samples,axis=0,ddof=1).max()<=cal['max_offset_se_mas']:kept[key]=(point-anchor,samples,n)
        names=sorted(kept);roles=[];vectors=[]
        for key in names:
            visit=key if key.startswith('NSC:') else None
            roles.append(bool(geometry.loc[visit,'role']) if visit in geometry.index else False)
            vectors.append(geometry.loc[visit,['zenith_0','zenith_1','zenith_2']].to_numpy(float) if visit in geometry.index else np.zeros(3))
        np.savez_compressed(work/('private_'+name+'.npz'),names=np.array(names),position=np.stack([kept[k][0] for k in names]),draws=np.stack([kept[k][1] for k in names]),counts=[kept[k][2] for k in names],gauge=np.repeat(metadata['reference_gauge'],len(names)),projection_center_deg=center,
            feature_coefficients=np.array([beta]),dcr_coefficient=np.array(beta),dcr_coefficient_draws=beta_draw-beta_draw.mean(),dcr_role=np.array(roles),zenith_vectors=np.array(vectors),reference_color_g_i=np.array(1.2),model_kind=np.array(name))
        models[name]=kept;coefficient[name]=dict(mas_per_color_tanz=beta,bootstrap_se=float(np.std(beta_draw,ddof=1)),includes_measured_reference_color_perturbations=enabled)
        color_info[name]=info;support[name]=dict(entered_groups=len(offset),retained_groups=len(kept),se_rejected_groups=len(offset)-len(kept))
    held=rows.loc[rows.is_holdout&rows.group.isin(models['D0_local_C0'])&rows.group.isin(models['D1_geometric_DCR'])].copy();residuals=[]
    for name in ('D0_local_C0','D1_geometric_DCR'):
        corrected=held[['east','north']].to_numpy()-np.array([models[name][k][0] for k in held.group])-coefficient[name]['mas_per_color_tanz']*np.nan_to_num(held.color.to_numpy()-1.2)[:,None]*held[['geometry_e','geometry_n']].to_numpy()
        residual=corrected-pd.DataFrame(corrected).groupby(held.reference.to_numpy()).transform('mean').to_numpy();residuals.append(residual)
    value=np.asarray(residuals);med=np.median(value,axis=1) if len(held) else None
    result=dict(preparation=metadata,models=coefficient,support=support,color_errors=color_info,common_held_measurements=len(held),common_held_references=held.reference.nunique(),held_demeaned_median_mas=med,held_demeaned_radial_p68_mas=np.quantile(np.linalg.norm(value,axis=2),.68,axis=1) if len(held) else None,
        slurm_job=os.environ['SLURM_JOB_ID'],adoption='not_adopted_pending_independent_QSO_blind_source_year_and_transfer_checks',
        scope='Same local reference graph, static nuisance positions, physical group zeros and scalar geometric coefficient. D0 no-DCR control uses identical centered sparse solver. Reference-object bootstrap and per-reference color perturbations are synchronous with group/coefficient draws. This propagates measured photometric errors, not unknown color-system or SED bias. Local model is not the original global C0; independent validation must retain that third baseline.')
    check_aggregate(result);save_json(work/'dcr_reference_summary.json',result)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('prepare','solve'));p.add_argument('work',type=Path);p.add_argument('--pilot',type=Path);p.add_argument('--reference',type=Path);p.add_argument('--sas',type=Path);p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use a Slurm compute node')
    a.work.mkdir(parents=True,exist_ok=True)
    if a.command=='prepare':
        if any(v is None for v in (a.pilot,a.reference,a.sas)):p.error('Preparation needs pilot, reference, sas')
        bind(a.work,'pilot_receipt.json',verify_directory(a.pilot,('private_fixed_qso.json.gz',)));bind(a.work,'reference_receipt.json',verify_directory(a.reference,('reference.csv.gz',)))
        with run_stage(a.work,PRECISION,'local_geometry','dcr-prepare-v15',inputs=('pilot_receipt.json','reference_receipt.json'),outputs=('dcr_preparation_summary.json','private_reference_dcr.csv.gz','private_color_photometry.csv','private_photo_file_receipts.json','private_group_geometry.csv')) as go:
            if go:prepare(a.work,a.pilot,a.reference,a.sas)
    else:
        with run_stage(a.work,PRECISION,'local_geometry','dcr-solve-v15',inputs=('dcr_preparation_summary.json','private_reference_dcr.csv.gz','private_color_photometry.csv','private_group_geometry.csv'),outputs=('dcr_reference_summary.json','private_D0_local_C0.npz','private_D1_geometric_DCR.npz')) as go:
            if go:solve(a.work,a.workers)


if __name__=='__main__':main()
