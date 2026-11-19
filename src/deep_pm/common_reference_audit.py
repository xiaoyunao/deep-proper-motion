"""Exact physical-group and held-object audit of original versus joint graphs."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .astrometry import project, unproject
from .continuous_reference import read_offsets, reference_frame, deduplicate, verify_directory
from .cross_region_audit import differences
from .io import digest,save_json
from .public_summary import check_aggregate


def mapped_points(table,path,center):
    solution,gauges=read_offsets(path,with_gauge=True)
    supported=table.group.isin(solution).to_numpy()
    pos=project(table.ra_deg,table.dec_deg,reference_frame(path))
    offsets=np.array([solution[k][0] if k in solution else [0.,0.] for k in table.group])
    return project(*unproject(pos-offsets,reference_frame(path)).T,center),supported,table.group.map(gauges)


def audit(state,joint,output):
    center=reference_frame(joint/'shared/private_c0.npz');joint_solution,joint_gauges=read_offsets(joint/'shared/private_c0.npz',with_gauge=True)
    verify_directory(joint/'shared',('private_c0.npz',))
    result={}
    for box in ('north_one','north_three'):
        parts=json.loads((state/f'references-{box}.json').read_text());frames=[]
        for source in parts.values():
            p=Path(source);verify_directory(p,('reference.csv.gz',))
            frames.append(pd.read_csv(p/'reference.csv.gz',dtype={'reference':str,'is_holdout':bool}))
        table,selection=deduplicate(frames)
        oldpath=state/'shared'/box/'private_c0.npz';old_solution,old_gauges=read_offsets(oldpath,with_gauge=True)
        mapping=Counter((old_gauges[k],joint_gauges[k]) for k in old_solution.keys()&joint_solution.keys())
        points=[];support=[];labels=[]
        for path in (oldpath,joint/'shared/private_c0.npz'):
            pos,valid,gauge=mapped_points(table,path,center);points.append(pos);support.append(valid);labels.append(gauge)
        use=table.is_holdout.to_numpy()&support[0]&support[1]
        held=table.loc[use,['reference','group','exposure_id']].copy()
        for j in range(2):
            held[['e'+str(j),'n'+str(j)]]=points[j][use]
            held['g'+str(j)]=labels[j][use].to_numpy()
        subsets={};measurements=points[1][use]-points[0][use]
        early=held.exposure_id.str.startswith('SDSS:').to_numpy()
        # Gauge pairs are explicit: never average unrelated component zero points.
        for gauge_pair, part in held.groupby(['g0','g1'],sort=True):
            medians=part.groupby(['reference',part.exposure_id.str.startswith('SDSS:')])[['e0','n0','e1','n1']].median().unstack()
            if True not in medians.columns.levels[1] or False not in medians.columns.levels[1]:continue
            delta=medians.xs(True,axis=1,level=1)-medians.xs(False,axis=1,level=1)
            delta=delta.dropna();before=delta[['e0','n0']].to_numpy();after=delta[['e1','n1']].to_numpy()
            subsets['old_'+str(gauge_pair[0])+'_joint_'+str(gauge_pair[1])]=dict(
                n=len(delta),before=differences(before),after=differences(after),change=differences(after-before))
        result[box]=dict(deduplication=selection,original_applied_gauges=dict(Counter(old_gauges.values())),
            joint_gauges_for_original_groups=dict(Counter(joint_gauges[k] for k in old_solution if k in joint_gauges)),
            old_to_joint_gauge_group_counts={str(a)+'_to_'+str(b):v for (a,b),v in mapping.items()},
            lost_original_group_count=len(old_solution.keys()-joint_solution.keys()),newly_supported_group_count=len((set(table.group)&joint_solution.keys())-old_solution.keys()),
            fixed_held_measurement_count=int(use.sum()),fixed_held_reference_count=held.reference.nunique(),
            held_corrected_position_change_mas=differences(measurements),
            held_change_by_survey={name:differences(measurements[early if name=='SDSS' else ~early]) for name in ('SDSS','NSC')},
            identical_held_source_cross_survey_mas=subsets,
            original_graph_sha256=digest(oldpath))
    public=dict(by_region=result,joint_graph_sha256=digest(joint/'shared/private_c0.npz'),slurm_job=os.environ['SLURM_JOB_ID'],
        scope='Same original held physical detections and identities in both graphs; exact inverse/forward coordinate conversion. Original and joint bootstrap draws are not paired; changes are descriptive and common zero errors are not divided by source count.')
    check_aggregate(public);save_json(output,public)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('state','joint','output'):p.add_argument(name,type=Path)
    a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use a Slurm compute node')
    audit(**vars(a))


if __name__=='__main__':main()
