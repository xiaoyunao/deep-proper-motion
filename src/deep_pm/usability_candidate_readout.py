"""Read the one completed selection trial without changing its decisions or fits."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .__main__ import tomllib
from .allband_qso import evaluated_fit
from .astrometry import project
from .blind_catalogue_quality import read
from .continuous_reference import PRECISION,SPATIAL,read_offsets,reference_frame
from .dark_precision import query_unique
from .exposure_catalogues import inside
from .io import digest,save_json
from .pilot_production import quality_detections
from .precision_v03 import calibration_groups
from .public_summary import check_aggregate
from .usability_cluster import summarize
from .validation import ratio


def one(task):
    row,trial=task;work=Path(trial)/row['core']
    config=tomllib.loads(PRECISION.read_text());settings=tomllib.loads(SPATIAL.read_text())
    graph=Path(row['graph'])/'private_c0.npz';center=reference_frame(graph)
    solution,gauges=read_offsets(graph,with_gauge=True)
    solution={k:v for k,v in solution.items() if gauges[k]==row['reference_gauge']}
    data,xy,time,sigma,keep=quality_detections(Path(row['source']),center,config,settings['production'])
    spectra=pd.read_csv(Path(row['source'])/'spectra_union.csv')
    spec=tomllib.loads(Path(row['geometry']).read_text())['fields'][row['core']]
    spectra=spectra.loc[inside(spectra.ra,spectra.dec,spec)].reset_index(drop=True)
    targets=project(spectra.ra,spectra.dec,center);tree=cKDTree(targets)
    keys=calibration_groups(data);lookup=pd.Index(data.detection_id);result=[]
    members=read(work/'private_full_candidate_members.json.gz')
    for row in members:
        ix=lookup.get_indexer(row['detection_ids'])
        if (ix<0).any():raise ValueError('Candidate member missing')
        position=np.median(xy[ix],axis=0)
        found,near=query_unique(tree,position.reshape(1,2),500,300)
        if not found[0]:continue
        k=int(near[0]);magnitude=float(spectra.reference_r.iat[k])
        if not 21<=magnitude<23:continue
        fit=evaluated_fit(ix,data,xy,time,sigma,keys,solution,center,config['reference_epoch'])
        result.append(dict(core=str(work.name),qso=k,magnitude=magnitude,motion=fit['pm_icrs_masyr'],error=fit['pm_error_masyr']))
    counts=Counter(r['qso'] for r in result)
    return [r for r in result if counts[r['qso']]==1]


def run(registration,trial,output,workers):
    plan=read(registration);trial_summary=read(trial/'v20_selection_candidate_summary.json')
    with ProcessPoolExecutor(max_workers=workers) as pool:rows=sum(pool.map(one,[(r,str(trial)) for r in plan['selected']]),[])
    cohorts={str(low):summarize([r['motion'] for r in rows if low<=r['magnitude']<low+1],
        [r['error'] for r in rows if low<=r['magnitude']<low+1]) for low in (21,22)}
    hold=[r['samples']['training']['held_year'] for r in trial_summary['cores']]
    n=sum(r['training_new_sources'] for r in hold);m=sum(r['uniquely_reproduced'] for r in hold)
    report=dict(trial_sha256=digest(trial/'v20_selection_candidate_summary.json'),added_qso=cohorts,
        total_training_candidates=n,unique_held_position_recovery=ratio(m,n),
        displaced_matches=[sum(r['shifted_counts'][j] for r in hold) for j in range(4)],
        unchanged_original_member_retention=ratio(trial_summary['counts']['full']['identical_members_retained'],trial_summary['counts']['full']['original']),
        adopted_dark_star_catalogue=False,
        scope='Added source identities fixed before this QSO readout. No second selection threshold or QSO-based source choice. Motion fitted from exact saved members; original catalogue unmodified. Held-year recovery is conditional on published retrieved detections, not independently certified stellar purity.',slurm_job=os.environ['SLURM_JOB_ID'])
    check_aggregate(report);save_json(output/'v20_selection_candidate_quality_summary.json',report)
    print('CANDIDATE_QUALITY_DONE',cohorts,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('registration','trial','output'):p.add_argument(n,type=Path)
    p.add_argument('--workers',type=int,default=2);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use Slurm compute')
    run(**vars(a))
