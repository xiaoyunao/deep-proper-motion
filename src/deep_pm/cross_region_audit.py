"""Empirical overlap audit of independently solved physical reference graphs.

Different graphs share galaxies and measurements but not bootstrap draws.
Report measured differences; never add marginal errors as if independent.
"""
import argparse
from collections import Counter
import gzip
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .astrometry import project,unproject
from .continuous_reference import verify_directory,read_offsets,reference_frame
from .dark_precision import query_unique
from .exposure_catalogues import coordinates
from .io import digest,save_json
from .public_summary import check_aggregate
from .standard_motion import icrs_motion


def frame_state(position,motion,center):
    point=project([position[0]],[position[1]],center)[0]
    jac=icrs_motion(point,np.zeros(2),center)['state_jacobian'][np.ix_([1,3],[1,3])]
    velocity=np.linalg.solve(jac,motion)
    return np.array([point[0],velocity[0],point[1],velocity[1]])


def differences(values):
    values=np.asarray(values)
    out=dict(n=len(values))
    if len(values)>=5:
        median=np.median(values,axis=0)
        out.update(median=median,nmad=1.4826*np.median(abs(values-median),axis=0),
                   radial_abs_p68=np.quantile(np.linalg.norm(values,axis=1),.68))
    return out


def match_rows(left,right):
    if not left or not right:return []
    a=np.array([r['position_deg'] for r in left]);b=np.array([r['position_deg'] for r in right])
    chord=lambda arcsec:2*np.sin(np.deg2rad(arcsec/3600)/2)
    good,indices=query_unique(cKDTree(coordinates(*b.T)),coordinates(*a.T),chord(1.),chord(.2))
    claims=Counter(indices[good])
    return [(left[i],right[indices[i]]) for i in np.flatnonzero(good) if claims[indices[i]]==1]


def source_check(left,right,standard):
    pairs=match_rows(left,right)
    values=[];shared=[];same=[];by_r={};unsupported_identity=0
    for a,b in pairs:
        if standard:
            x,y=map(set,(a['detection_ids'],b['detection_ids']))
            if len(x&y)<2:
                unsupported_identity+=1
                continue
        ma=a['pm_icrs_masyr'] if standard else a['motion_icrs']['C0']
        mb=b['pm_icrs_masyr'] if standard else b['motion_icrs']['C0']
        center=tuple(a['position_deg'])
        delta=frame_state(b['position_deg'],mb,center)-frame_state(a['position_deg'],ma,center)
        values.append(delta)
        tag=a.get('r_bin','unknown') if standard else str(int(a['magnitude']))
        by_r.setdefault(tag,[]).append(delta[[1,3]])
        if standard:
            shared.append(len(x&y));same.append(x==y)
    out=dict(unique_common_count=len(values),position_pairs_without_shared_identity_support=unsupported_identity,
        motion_difference_masyr=differences([v[[1,3]] for v in values]),
        position_difference_mas=differences([v[[0,2]] for v in values]),
        by_r={tag:differences(v) for tag,v in by_r.items()})
    if standard:
        out.update(identical_detection_sets=sum(same),shared_detection_p16_p50_p84=np.quantile(shared,[.16,.5,.84]) if shared else None,
            identical_detection_subset_motion_difference_masyr=differences([v[[1,3]] for v,identical in zip(values,same) if identical]),
            identical_detection_subset_position_difference_mas=differences([v[[0,2]] for v,identical in zip(values,same) if identical]),
            changed_detection_subset_motion_difference_masyr=differences([v[[1,3]] for v,identical in zip(values,same) if not identical]))
    return out


def galaxy_check(left,right,graphs,center,selected_gauges):
    tables=[]
    for path in (left,right):
        verify_directory(path,('reference.csv.gz','reference_summary.json'))
        table=pd.read_csv(path/'reference.csv.gz')
        if table.detection_id.duplicated().any():raise ValueError('Duplicate physical reference detections')
        tables.append(table)
    merged=tables[0].merge(tables[1],on='detection_id',suffixes=('_a','_b'),validate='one_to_one')
    held=merged.is_holdout_a.eq(True)&merged.is_holdout_b.eq(True)
    supported=np.ones(len(merged),bool);corrected=[]
    for suffix,graph,selected in zip(('a','b'),graphs,selected_gauges):
        offsets,gauges=read_offsets(graph,with_gauge=True);frame=reference_frame(graph)
        offsets={key:value for key,value in offsets.items() if gauges[key]==selected}
        supported &= merged['group_'+suffix].isin(offsets).to_numpy()
        sky=merged[['ra_deg_'+suffix,'dec_deg_'+suffix]].to_numpy()
        xy=project(*sky.T,frame)
        delta=np.array([offsets[k][0] if k in offsets else [0.,0.] for k in merged['group_'+suffix]]).reshape(-1,2)
        corrected.append(project(*unproject(xy-delta,frame).T,center))
    np.testing.assert_allclose(merged[['ra_deg_a','dec_deg_a']].to_numpy(),merged[['ra_deg_b','dec_deg_b']].to_numpy(),atol=1e-10,rtol=0)
    delta=corrected[1]-corrected[0]
    use=held.to_numpy()&supported
    common=set(merged.group_a)&set(merged.group_b)
    return dict(shared_physical_group_count=len(common),common_detection_count=len(merged),
        both_held_supported_detection_count=int(use.sum()),
        both_held_reference_pair_count=int(merged.loc[use,['reference_a','reference_b']].drop_duplicates().shape[0]),
        held_position_difference_mas=differences(delta[use]),
        by_survey={survey:differences(delta[use&merged.group_a.str.startswith(survey+':').to_numpy()]) for survey in ('SDSS','NSC')},
        interpretation='Same physical held measurements corrected by two graphs. Descriptive shared-error difference, not independent astrometric accuracy.')


def audit(left,right,left_ref,right_ref,left_graph,right_graph,output):
    graphs=[p/'private_c0.npz' for p in (left_graph,right_graph)]
    for p in (left_graph,right_graph):verify_directory(p,('private_c0.npz','continuous_reference_summary.json'))
    standard=[];qso=[];gauges=[]
    for p in (left,right):
        verify_directory(p,('private_standard_motion.json.gz','private_qso_blocks.json','continuous_core_summary.json'))
        with gzip.open(p/'private_standard_motion.json.gz','rt') as f:standard.append(json.load(f))
        qso.append(json.loads((p/'private_qso_blocks.json').read_text()))
        gauges.append(json.loads((p/'continuous_core_summary.json').read_text())['reference_gauge'])
    center=tuple(np.mean([reference_frame(p) for p in graphs],axis=0))
    result=dict(stars=source_check(*standard,True),spectral_qso=source_check(*qso,False),
        held_galaxies=galaxy_check(left_ref,right_ref,graphs,center,gauges),
        applied_reference_components=gauges,
        graph_sha256=[digest(p) for p in graphs],projection_centers_deg=[reference_frame(p) for p in graphs],
        covariance_status='Cross-graph galaxy/measurement covariance not solved. Do not normalize differences with independent quadrature or pair bootstrap draws.',
        matching_scope='Star pairs require unique position and at least two shared physical detections. Spectral pairs are stationary-QSO position controls. No blind-member purity claim.',
        spatial_zero_status='not_certified',slurm_job=os.environ['SLURM_JOB_ID'])
    check_aggregate(result);save_json(output,result)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('left','right','left_ref','right_ref','left_graph','right_graph','output'):p.add_argument(name,type=Path)
    a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use a Slurm compute node')
    audit(**vars(a))


if __name__=='__main__':main()
