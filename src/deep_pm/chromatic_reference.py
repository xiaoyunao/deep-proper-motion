"""One identifiable residual DECam-g color term jointly with physical C0 zeros.

The main northern component is a research trial. The independent component
keeps the frozen C0 solution. No target QSO or stellar proper motion trains it.
Source and calibrator color uncertainties remain a qualification limitation.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .__main__ import tomllib
from .continuous_instruments import chip_receipts,instrument_by_exposure
from .continuous_reference import PRECISION,bind,read_offsets,reference_frame,verify_directory,deduplicate,calibration_components
from .io import save_json,digest
from .joint_reference import identity_split
from .provenance import run_stage
from .public_summary import check_aggregate
from .zero_point import fit_offsets

POLICY=dict(component='0',instrument='c4d',band='g',reference_color_g_i=1.2,unknown_color_feature=0.,
    target_motion_or_qso_training=False,other_components='frozen_C0',other_bands='no_extra_color_coefficient',
    uncertainty='Synchronous reference-object bootstrap for group zeros and color coefficient. Colors treated fixed; no production adoption without color-error qualification and independent validation.')


def prepare(work,state,joint,workers):
    geometry=joint/'joint_geometry.toml';plan=json.loads((joint/'private_joint_reference_plan.json').read_text())
    def shard(item):
        name,ref_path=item;ref=Path(ref_path);source=state/'inputs'/name;target=work/'metadata'/name;target.mkdir(parents=True,exist_ok=True)
        bind(target,'reference_receipt.json',verify_directory(ref,('reference.csv.gz',)))
        bind(target,'source_receipt.json',verify_directory(source,('detections.csv',)))
        with run_stage(target,geometry,name,'chromatic-reference-metadata-v14',inputs=('reference_receipt.json','source_receipt.json'),outputs=('private_sdss_colors.csv','private_group_features.csv','feature_metadata_summary.json')) as go:
            if go:
                frame=pd.read_csv(ref/'reference.csv.gz',dtype={'reference':str,'is_holdout':bool})
                selected=set(frame.loc[frame.exposure_id.str.startswith('SDSS:'),'detection_id'])
                chips,complete=chip_receipts((source/'run_manifest.json').resolve());instrument=instrument_by_exposure(chips)
                pieces=[];groups={}
                columns=pd.read_csv(source/'detections.csv',nrows=0).columns
                wanted=[k for k in ('survey','detection_id','exposure_id','band','color') if k in columns]
                for chunk in pd.read_csv(source/'detections.csv',usecols=wanted,chunksize=200000,low_memory=False):
                    early=chunk.loc[chunk.detection_id.isin(selected),['detection_id']].copy()
                    early['color']=pd.to_numeric(chunk.loc[early.index,'color'],errors='coerce') if 'color' in chunk else np.nan
                    pieces.append(early)
                    late=chunk.loc[chunk.survey.eq('NSC'),['exposure_id','band']].drop_duplicates()
                    for exposure,band in late.itertuples(index=False,name=None):
                        value=(str(band),instrument.get(exposure,'unknown'))
                        if groups.setdefault(exposure,value)!=value:raise ValueError('Conflicting metadata for one physical group')
                colors=pd.concat(pieces,ignore_index=True) if pieces else pd.DataFrame(columns=['detection_id','color'])
                if colors.detection_id.duplicated().any():raise ValueError('Repeated physical SDSS color in one input')
                colors.to_csv(target/'private_sdss_colors.csv',index=False)
                group_data=pd.DataFrame([dict(group=k,band=v[0],instrument=v[1],feature_g=v==('g','c4d')) for k,v in groups.items()],columns=['group','band','instrument','feature_g'])
                group_data.to_csv(target/'private_group_features.csv',index=False)
                result=dict(reference_identity_split_sha256=identity_split(frame),sdss_color_measurements=len(colors),physical_groups=len(groups),chip_receipts_complete=complete)
                check_aggregate(result);save_json(target/'feature_metadata_summary.json',result)
        return name,str(target.absolute())
    with ThreadPoolExecutor(max_workers=workers) as pool:parts=dict(pool.map(shard,plan.items()))
    save_json(work/'private_metadata_plan.json',parts)


def assemble(work,joint):
    original=json.loads((joint/'private_joint_reference_plan.json').read_text());metadata=json.loads((work/'private_metadata_plan.json').read_text())
    frames=[];colors=[];groups=[]
    for name,path in original.items():
        part=Path(metadata[name]);verify_directory(part,('private_sdss_colors.csv','private_group_features.csv','feature_metadata_summary.json'))
        frame=pd.read_csv(Path(path)/'reference.csv.gz',dtype={'reference':str,'is_holdout':bool})
        if identity_split(frame)!=json.loads((part/'feature_metadata_summary.json').read_text())['reference_identity_split_sha256']:raise ValueError('Frozen reference identity changed')
        frames.append(frame);colors.append(pd.read_csv(part/'private_sdss_colors.csv',dtype={'detection_id':str}));groups.append(pd.read_csv(part/'private_group_features.csv',dtype={'feature_g':bool}))
    rows,selection=deduplicate(frames)
    color=pd.concat(colors,ignore_index=True)
    spread=color.groupby('detection_id').color.agg(['min','max'])
    if ((spread['max']-spread['min'])>1e-7).any():raise ValueError('Same physical detection has inconsistent colors')
    color=color.drop_duplicates('detection_id').set_index('detection_id').color
    group=pd.concat(groups,ignore_index=True)
    if group.groupby('group').feature_g.nunique().gt(1).any():raise ValueError('Physical feature roles differ between regions')
    role=group.drop_duplicates('group').set_index('group').feature_g
    early=rows.loc[rows.exposure_id.str.startswith('SDSS:'),['reference','detection_id']].copy();early['color']=early.detection_id.map(color)
    early.loc[~early.color.between(-1,4,inclusive='neither'),'color']=np.nan
    reference_color=early.groupby('reference').color.median();rows['color']=rows.reference.map(reference_color)
    rows['feature_role']=rows.group.map(role).fillna(False).astype(bool)
    rows['feature']=np.where(rows.feature_role&rows.color.notna(),rows.color-POLICY['reference_color_g_i'],0.)
    return rows,selection


def solve(work,joint):
    rows,selection=assemble(work,joint);config=tomllib.loads(PRECISION.read_text());cal=config['calibration'];train=rows.loc[~rows.is_holdout]
    original=joint/'shared/private_c0.npz';base,gauges=read_offsets(original,with_gauge=True)
    coefficients={k:np.zeros(2) for k in set(gauges.values())};feature_draws={k:np.zeros((cal['bootstrap_draws'],2)) for k in coefficients}
    offset={k:v for k,v in base.items() if gauges[k]!=POLICY['component']};new_gauges={k:gauges[k] for k in offset}
    for component,part in enumerate(calibration_components(train,cal['min_references'])):
        if str(component)!=POLICY['component']:continue
        fitted,point,draw=fit_offsets(part.reference,part[['east','north']].to_numpy(),part[['sigma_e','sigma_n']].to_numpy(),part.group.to_numpy(),cal['min_references'],cal['bootstrap_draws'],config['seed']+component,workers=int(os.environ.get('DEEP_PM_M3_WORKERS','1')),features=part[['feature']].to_numpy())
        late=[v for k,v in fitted.items() if k.startswith('NSC:')];anchor=np.median(np.stack([v[0] for v in late]),axis=0)
        datum=np.median(np.stack([v[1]-v[1].mean(axis=0) for v in late]),axis=0)
        coefficients[str(component)]=point[0];feature_draws[str(component)]=draw[:,0]-draw[:,0].mean(axis=0)
        for key,(position,samples,n) in fitted.items():
            centered=samples-samples.mean(axis=0)-datum
            if np.std(centered,axis=0,ddof=1).max()<=cal['max_offset_se_mas']:
                offset[key]=(position-anchor,centered,n);new_gauges[key]=str(component)
    if not any(new_gauges[k]==POLICY['component'] for k in offset):raise ValueError('No chromatic reference component fitted')
    names=sorted(offset);components=sorted(coefficients);roles=rows.groupby('group').feature_role.first().to_dict()
    np.savez_compressed(work/'private_chromatic_reference.npz',names=np.array(names),position=np.stack([offset[k][0] for k in names]),draws=np.stack([offset[k][1] for k in names]),counts=np.array([offset[k][2] for k in names]),gauge=np.array([new_gauges[k] for k in names]),projection_center_deg=reference_frame(original),
        feature_component_labels=np.array(components),feature_coefficients=np.stack([coefficients[k] for k in components]),feature_draws=np.stack([feature_draws[k] for k in components]),feature_groups=np.array([bool(roles.get(k,False)) and new_gauges[k]==POLICY['component'] for k in names]),reference_color_g_i=np.array(POLICY['reference_color_g_i']))
    held=rows.loc[rows.is_holdout&rows.group.isin(offset)&rows.group.isin(base)].copy()
    held=held.loc[held.group.map(new_gauges)==held.group.map(gauges)]
    xy=held[['east','north']].to_numpy();old=xy-np.array([base[k][0] for k in held.group])
    group_beta={k:coefficients[new_gauges[k]] if roles.get(k,False) else np.zeros(2) for k in offset}
    color=np.nan_to_num(held.color.to_numpy()-POLICY['reference_color_g_i'],nan=0.)
    new=xy-np.array([offset[k][0] for k in held.group])-np.array([group_beta[k] for k in held.group])*color[:,None]
    held[['old_e','old_n']]=old;held[['new_e','new_n']]=new
    held['is_sdss']=held.exposure_id.str.startswith('SDSS:');held['applied_gauge']=held.group.map(new_gauges)
    comparisons={}
    for name,selected in [('cross_survey',held),('c4d_g_to_sdss',held.loc[held.is_sdss|held.feature_role])]:
        med=selected.groupby(['reference','applied_gauge','is_sdss'])[['old_e','old_n','new_e','new_n']].median().unstack('is_sdss')
        if False not in med.columns.levels[-1] or True not in med.columns.levels[-1]:continue
        delta=(med.xs(False,axis=1,level=-1)-med.xs(True,axis=1,level=-1)).dropna().to_numpy().reshape(-1,2,2)
        comparisons[name]=dict(n=len(delta),median_mas=np.median(delta,axis=0),radial_p68_mas=np.quantile(np.linalg.norm(delta,axis=2),.68,axis=0))
    result=dict(reference_selection=selection,preserved_reference_identity_split_sha256=identity_split(rows),
        training_reference_count=train.reference.nunique(),known_color_training_references=train.loc[train.color.notna()].reference.nunique(),
        feature_policy=POLICY,feature_coefficients_mas_per_color=coefficients,feature_bootstrap_se_mas_per_color={k:np.std(v,axis=0,ddof=1) for k,v in feature_draws.items()},
        retained_groups=len(offset),lost_frozen_groups=len(base.keys()-offset.keys()),new_supported_groups=len(offset.keys()-base.keys()),
        held_galaxy_fixed_comparisons=comparisons,
        slurm_job=os.environ['SLURM_JOB_ID'],adoption='research_only_no_production_without_independent_QSO_and_year_validation')
    check_aggregate(result);save_json(work/'chromatic_reference_summary.json',result)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('prepare','solve'));p.add_argument('work',type=Path);p.add_argument('state',type=Path);p.add_argument('joint',type=Path);p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use a Slurm compute node')
    a.work.mkdir(parents=True,exist_ok=True);policy=a.work/'feature_policy.json'
    if policy.exists() and json.loads(policy.read_text())!=POLICY:raise ValueError('Chromatic policy changed')
    if not policy.exists():save_json(policy,POLICY)
    if a.command=='prepare':prepare(a.work,a.state,a.joint,a.workers);return
    bind(a.work,'original_joint_reference_receipt.json',verify_directory(a.joint/'shared',('private_c0.npz',)))
    inputs=['feature_policy.json','original_joint_reference_receipt.json','private_metadata_plan.json']
    for name,path in json.loads((a.work/'private_metadata_plan.json').read_text()).items():
        alias=name+'_metadata_receipt.json';bind(a.work,alias,Path(path)/'run_manifest.json');inputs.append(alias)
    with run_stage(a.work,a.joint/'joint_geometry.toml','north_main_g_color','chromatic-reference-v14',inputs=inputs,outputs=('private_chromatic_reference.npz','chromatic_reference_summary.json')) as go:
        if go:solve(a.work,a.joint)


if __name__=='__main__':main()
