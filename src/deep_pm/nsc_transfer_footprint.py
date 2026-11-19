"""Measure coverage from real calibrated CCD and SDSS scan footprints."""
import argparse
import json
import os
from pathlib import Path

from astropy.io import fits
import numpy as np
import pandas as pd

from . import access
from .__main__ import tomllib
from .astrometry import sdss_tai_to_tt, years
from .continuous_reference import bind, read_offsets, verify_directory
from .continuous_summary import area
from .exposure_preflight import where_box
from .footprints import sdss_window_mask, sdss_window_selection
from .io import digest, save_json
from .nsc_exposure_complete import CHIP_VERTICES
from .nsc_opportunity_area import estimate
from .provenance import run_stage
from .public_summary import check_aggregate


def run(work,plan,geometry,shared,window):
    data = json.loads(plan.read_text())
    graph = shared/'private_c0.npz'
    if data['reference_sha256'] != digest(graph) or data['geometry_sha256']!=digest(geometry):
        raise ValueError('Different frozen footprint geometry or graph')
    specs = tomllib.loads(geometry.read_text())['fields']
    domain = [specs[k] for k in data['cores']]
    box = {key:[min(p[key][0] for p in domain),max(p[key][1] for p in domain)] for key in ('ra','dec')}
    center = [float(np.mean(box[k])) for k in ('ra','dec')]
    solution,gauges = read_offsets(graph,with_gauge=True)
    if len(set(gauges.values()))!=1:
        raise ValueError('Opportunity area needs a single compatible graph gauge')
    pad=.7
    chips=access.checked_spatial(access.Cache(work/'chip_cache'),'nsc',
        ','.join(['exposure','ccdnum','instrument','filter','mjd']+CHIP_VERTICES),'nsc_dr2.chip',
        where_box(box,pad),box['dec'][0]-pad-1e-7,box['dec'][1]+pad+1e-7).drop_duplicates()
    if chips.duplicated(['exposure','ccdnum']).any():
        raise ValueError('Conflicting physical chip metadata')
    chips.to_csv(work/'private_transfer_chips.csv.gz',index=False)
    with fits.open(window,memmap=True) as handle:
        table=handle[1].data
        selected=sdss_window_mask(table,301)&sdss_window_selection(table,center,8.)
        columns=['RA','DEC','RUN','CAMCOL','NODE','INCL','MU_START','MU_END','NU_START','NU_END']
        frame=pd.DataFrame({k:np.asarray(table[k][selected],float) for k in columns})
        tai=np.asarray(table['TAI'][selected],float)
        if tai.ndim==2:
            tai=tai[:,2]
        frame['epoch']=years(sdss_tai_to_tt(np.where((tai>=4e9)&(tai<6e9),tai,np.nan)))
        frame=frame.loc[np.isfinite(frame.epoch)]
    grids={str(step):estimate(chips,frame,box,solution,step,domain) for step in (.05,.025)}
    result=dict(area_sqdeg=sum(area(x) for x in domain),cores=len(domain),grids=grids,
                plan_sha256=digest(plan),reference_sha256=digest(graph),chip_sha256=digest(work/'private_transfer_chips.csv.gz'),
                slurm_job=os.environ['SLURM_JOB_ID'],
                scope='Footprint-only opportunity inside the 72 registered half-open cores. SDSS and NSC physical C0 support, at least two NSC exposures, NSC span >= 0.60 yr and joint span >= 5 yr. This does not count per-target r/other-band depth, completeness, reliable source identities or certified area.')
    check_aggregate(result)
    save_json(work/'transfer_footprint_summary.json',result)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('work','plan','geometry','shared','window'):
        p.add_argument(name,type=Path)
    a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        p.error('Use Slurm')
    a.work.mkdir(parents=True,exist_ok=True)
    for name,path in [('frozen_plan.json',a.plan),('window.fits',a.window),
                      ('reference_receipt.json',verify_directory(a.shared,('private_c0.npz',)))]:
        bind(a.work,name,path)
    with run_stage(a.work,a.geometry,'north_two','nsc-transfer-footprint-v16',
                   ('frozen_plan.json','window.fits','reference_receipt.json'),
                   ('private_transfer_chips.csv.gz','transfer_footprint_summary.json')) as go:
        if go:run(**vars(a))


if __name__ == '__main__':
    main()
