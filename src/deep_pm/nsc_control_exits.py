"""Trace old r controls through the same physical all-band selection funnel."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .__main__ import tomllib
from .allband_qso import physical_candidates
from .astrometry import unproject
from .astrometry_audit import augment_tracks
from .continuous_reference import PRECISION, SPATIAL, bind, read_offsets, reference_frame, verify_directory
from .exposure_catalogues import inside
from .io import save_json
from .pilot_production import make_tracks, quality_detections, source_funnel
from .precision_v03 import calibration_groups, calibration_support
from .provenance import run_stage
from .public_summary import check_aggregate


def run(work, state, audit, geometry, core):
    targets = [r for r in json.loads((audit/'private_unmatched_controls.json').read_text()) if r['core']==core]
    folder = Path(json.loads((state/'fits.json').read_text())[core])
    graph = (folder/'reference_receipt.json').resolve().parent/'private_c0.npz'
    source = (folder/'source_receipt.json').resolve().parent
    verify_directory(source,('detections.csv',))
    spec = tomllib.loads(geometry.read_text())['fields'][core]
    gauge = json.loads((folder/'nsc_science_catalogue_summary.json').read_text())['reference_gauge']
    solution,gauges = read_offsets(graph,with_gauge=True)
    solution = {k:v for k,v in solution.items() if gauges[k]==gauge}
    center = reference_frame(graph)
    config = tomllib.loads(PRECISION.read_text())
    settings = tomllib.loads(SPATIAL.read_text())
    prod = settings['production']
    data,xy,time,sigma,keep = quality_detections(source,center,config,prod)
    keys = calibration_groups(data)
    supports = calibration_support(keys,solution)
    visit = data.exposure_id.to_numpy()
    nsc = data.survey.eq('NSC').to_numpy()
    physical = physical_candidates(data)
    strict = keep & (~nsc | physical)
    full = strict | (nsc & physical & supports)
    seed = make_tracks(xy,visit,strict & nsc,prod['match_radius_mas'])
    tracks = augment_tracks(make_tracks(xy,visit,strict,prod['match_radius_mas']),data,xy,full,strict,prod)
    index = pd.Index(data.detection_id)
    relevant = set(index.get_indexer([item for row in targets for item in row['detection_ids']]))
    if -1 in relevant:
        raise ValueError('Lost old control measurements')
    seeds = [track for track in seed if relevant.intersection(track)]
    trace = {}
    owns = lambda xy: inside(*unproject(xy,center)[0],spec)
    source_funnel(data,xy,time,sigma,full,tracks,solution,config,settings,9.,
        reference_keep=strict,trace=trace,initial=seeds,observation_groups=keys,core_members=owns)
    ownership = {}
    for j,track in enumerate(tracks):
        for i in relevant.intersection(track):
            ownership[i] = j
    counts = Counter()
    private = []
    for target in targets:
        rows = index.get_indexer(target['detection_ids'])
        votes = Counter(ownership[i] for i in rows if i in ownership)
        chosen = votes.most_common(1)
        if chosen:
            j,shared = chosen[0]
            reason = trace.get(j,'no_overlapping_r_seed')
            if reason=='final' and shared<.6*len(rows):
                reason = 'final_fit_below_frozen_member_overlap'
            elif reason=='final' and len(votes)>1:
                reason = 'final_fit_with_competing_track'
            elif reason=='final':
                reason = 'final_fit_claimed_another_old_control'
        else:
            reason = 'no_retained_physical_track'
            shared = 0
        counts[reason] += 1
        private.append(dict(control_slot=target['slot'],stages={str(j):trace.get(j,'no_overlapping_r_seed') for j in votes},
                            shared_members_of_best_track=shared,old_members=len(rows),
                            physical_nsc_members_removed=int((nsc[rows]&~physical[rows]).sum()),
                            has_catalogue_member_overlap=bool(target['member_overlap']),status=reason))
    out = dict(core=core, old_controls=len(targets), furthest_trace_counts=dict(counts),
                any_physical_nsc_members_removed=sum(r['physical_nsc_members_removed']>0 for r in private),
                any_competing_track=sum(len(r['stages'])>1 for r in private),
                slurm_job=os.environ['SLURM_JOB_ID'],
                scope='Exact v1.5 physical quality, tracking and funnel with old-member-touching r seeds only. Status follows the furthest stage of the maximum-overlap track. It does not establish physical truth. No motion-based identity match.')
    check_aggregate(out)
    save_json(work/'control_exit_summary.json',out)
    save_json(work/'private_control_exit_trace.json',private)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('work','state','audit','geometry'):
        p.add_argument(name,type=Path)
    p.add_argument('core')
    a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        p.error('Use Slurm')
    a.work.mkdir(parents=True,exist_ok=True)
    bind(a.work,'state_receipt.json',verify_directory(a.state,('fits.json','private_owner_overlay.json')))
    bind(a.work,'audit_receipt.json',verify_directory(a.audit,('private_unmatched_controls.json',)))
    with run_stage(a.work,a.geometry,a.core,'nsc-control-exits-v16',('state_receipt.json','audit_receipt.json'),
                   ('control_exit_summary.json','private_control_exit_trace.json')) as go:
        if go:
            run(**vars(a))


if __name__=='__main__':
    main()
