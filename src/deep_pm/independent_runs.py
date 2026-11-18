"""Lock a first acquisition batch using geometry and conservative prior run reuse."""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from .astrometry import project
from .footprints import read_sdss_window,sdss_visits
from .independent_selection import previous_footprints
from .io import digest,save_json
from .public_summary import check_aggregate


def ranked_first_batch(rows):
    first=[]
    for box in sorted(set(r['box'] for r in rows)):
        candidates=[r for r in rows if r['box']==box and r['selected_for_reference_precheck']]
        if candidates:
            first.append(min(candidates,key=lambda r:(r['new_sdss_runs']==0,r['shared_prior_sdss_runs'],
                -r['joint_timed_fraction'],-r['spectroscopic_qso_21_23_count'],r['name'])))
    return first


def run(geometry,window_path,work):
    data=json.loads(geometry.read_text())
    window=read_sdss_window(window_path,301)
    prior=set()
    for old in previous_footprints():
        center=tuple(float(np.mean(old[k])) for k in ('ra','dec'))
        radius=old.get('radius',np.hypot(np.ptp(np.atleast_1d(old['ra']))/2,np.ptp(np.atleast_1d(old['dec']))/2))+.06
        # Include every metadata-preselected run; conservative about independence.
        prior.update(v.exposure for v in sdss_visits(None,center,radius,rerun=301,vectorized=True,window=window))
    rng=np.random.default_rng(9137)
    inventory={}
    for row in data['candidates']:
        if not row['selected_for_reference_precheck']:
            continue
        ra,dec=row['ra_bounds_deg'],row['dec_bounds_deg']
        center=float(np.mean(ra)),float(np.mean(dec))
        sky=np.column_stack([rng.uniform(*ra,512),np.degrees(np.arcsin(rng.uniform(*np.sin(np.radians(dec)),512)))])
        xy=project(*sky.T,center)
        runs={v.exposure for v in sdss_visits(None,center,1.6,rerun=301,vectorized=True,window=window) if v.coverage(xy,sky).any()}
        inventory[row['name']]=sorted(runs)
        row.update(physical_sdss_runs=len(runs),shared_prior_sdss_runs=len(runs&prior),new_sdss_runs=len(runs-prior))
    selected=ranked_first_batch([r for r in data['candidates'] if r['selected_for_reference_precheck']])
    work.mkdir(parents=True,exist_ok=True)
    save_json(work/'private_run_inventory.json',dict(prior=sorted(prior),candidates=inventory))
    (work/'first-batch.tsv').write_text(''.join(f"{r['name']}\t{r['box']}\n" for r in selected))
    result=dict(geometry_summary_sha256=digest(geometry),sdss_window_sha256=digest(window_path),
        prior_run_count=len(prior),first_batch=selected,first_batch_area_sqdeg=sum(r['area_sqdeg'] for r in selected),
        candidate_new_area_sqdeg=data['provisional_new_area_sqdeg'],selection_uses_proper_motion=False,
        run_independence_scope='Prior union is conservative metadata preselection; candidate coverage sampled at 512 positions. Verify actual used runs after acquisition.',
        reference_support='Pending first-batch galaxy graph verification.',slurm_job=os.environ['SLURM_JOB_ID'])
    check_aggregate(result)
    save_json(work/'first_batch_selection_summary.json',result)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('geometry','window','work'):
        p.add_argument(name,type=Path)
    a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        p.error('Use a Slurm compute node')
    run(a.geometry,a.window,a.work)


if __name__=='__main__':
    main()
