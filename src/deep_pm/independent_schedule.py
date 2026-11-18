"""Extend a registered cumulative batch in geometry-only deterministic order."""
import argparse
import json
import os
from pathlib import Path

from .io import digest,save_json
from .public_summary import check_aggregate


def extend(geometry,previous,count):
    old=previous['first_batch']
    names={r['name'] for r in old}
    eligible=[r for r in geometry['candidates'] if r['selected_for_reference_precheck']]
    registered={r['name']:r for r in eligible}
    if not names<=registered.keys():raise ValueError('Previous batch is outside registered geometry')
    new=[]
    for box in sorted({r['box'] for r in eligible}):
        choices=sorted((r for r in eligible if r['box']==box and r['name'] not in names),key=lambda r:r['name'])
        new.extend(choices[:count])
    cumulative=[registered[r['name']] for r in old]+new
    return dict(first_batch=cumulative,new_parent_count=len(new),cumulative_parent_count=len(cumulative),
                first_batch_area_sqdeg=sum(r['area_sqdeg'] for r in cumulative),
                remaining_parent_count=len(eligible)-len(cumulative),selection_uses_proper_motion=False,
                selection_rule='Up to the declared count of remaining registered parents per window, lexicographic name order. No new motion outcomes read.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('geometry','previous','output'):p.add_argument(key,type=Path)
    p.add_argument('--per-window',type=int,default=4)
    a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use Slurm compute allocation')
    if a.per_window<1:p.error('Positive batch size required')
    previous=json.loads(a.previous.read_text())
    if previous['geometry_summary_sha256']!=digest(a.geometry):raise ValueError('Registered geometry changed')
    result=extend(json.loads(a.geometry.read_text()),previous,a.per_window)
    result.update(geometry_summary_sha256=digest(a.geometry),previous_selection_sha256=digest(a.previous),
                  slurm_job=os.environ['SLURM_JOB_ID'])
    check_aggregate(result)
    save_json(a.output,result)


if __name__=='__main__':main()
