"""Freeze whole-C input opportunities and estimate all-object retrieval resources."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
from astropy.time import Time

from . import access
from .__main__ import tomllib
from .continuous_reference import bind,verify_directory,read_offsets
from .exposure_inputs import tap_rectangle
from .exposure_preflight import where_box
from .io import digest,save_json
from .provenance import run_stage
from .public_summary import check_aggregate
from .regional_preflight import CHIP_VERTICES


def run(work,state,geometry):
    config=tomllib.loads(geometry.read_text());box=config['boxes']['c'];halo=float(config['halo_deg']);cache=access.Cache(work/'cache')
    plans={k:json.loads((state/(k+'.json')).read_text()) for k in ('fits','sources','references')}
    if not set(plans['fits'])<=set(plans['sources']):raise ValueError('Frozen source plan incomplete')
    for name in ('fits','sources','references'):save_json(work/('private_frozen_'+name+'.json'),plans[name])
    where=where_box(box,halo);counts={}
    for name,selection in [('all_objects',''),('legacy_r_index',' AND ndetr>0 AND (rmag<24 OR nphotr=0 OR rmag IS NULL)')]:
        query='SELECT COUNT(*) AS n,SUM(ndet) AS reported_measurements FROM nsc_dr2.object WHERE '+where+selection
        value=cache.query('nsc',query,expected=1).iloc[0];counts[name]=dict(objects=int(value['n']),reported_measurements=int(value['reported_measurements']))
    pad=.65;metadata_where=where_box(box,halo+pad);lo,hi=box['dec']
    cols=['exposure','ccdnum','ra','dec','instrument','filter','mjd','airmass']+CHIP_VERTICES
    chips=access.checked_spatial(cache,'nsc',','.join(cols),'nsc_dr2.chip',metadata_where,lo-halo-pad,hi+halo+pad)
    if chips.duplicated(['exposure','ccdnum']).any():raise ValueError('Duplicate physical chip metadata')
    chips.to_csv(work/'private_continuous_chip_opportunities.csv.gz',index=False)
    # Bounding-box intersection is deliberately conservative here. Exact CCD
    # masks and qualified target measurements will define usable scientific area.
    center=float(np.mean(box['ra']));longitude=np.column_stack([(chips['vertex_ra'+str(i)].to_numpy()-center+180)%360-180+center for i in range(1,5)])
    latitude=chips[['vertex_dec'+str(i) for i in range(1,5)]].to_numpy()
    use=(longitude.max(axis=1)>=box['ra'][0])&(longitude.min(axis=1)<=box['ra'][1])&(latitude.max(axis=1)>=lo)&(latitude.min(axis=1)<=hi)
    selected=chips.loc[use].copy();selected['night_utc18']=np.floor(selected.mjd-.75).astype(int);selected['year']=np.array([int(v[:4]) for v in Time(selected.mjd.to_numpy(),format='mjd',scale='utc').isot])
    reference=state/'shared/private_c0.npz';solution=read_offsets(reference);selected['original_reference_support']=('NSC:'+selected.exposure.astype(str)).isin(solution)
    bands={}
    for (instrument,band),part in selected.groupby(['instrument','filter'],sort=True):
        bands[str(instrument)+':'+str(band)]=dict(exposures=part.exposure.nunique(),ccds=len(part),nights=part.night_utc18.nunique(),years=sorted(map(int,set(part.year))),original_c0_supported_exposures=part.loc[part.original_reference_support,'exposure'].nunique())
    try:
        quota=subprocess.run(['lfs','quota','-u',os.environ['USER'],str(state)],capture_output=True,text=True,timeout=30);(work/'private_quota.txt').write_text(quota.stdout+quota.stderr)
    except (OSError,subprocess.TimeoutExpired):pass
    area=np.deg2rad(np.ptp(box['ra']))*np.diff(np.sin(np.deg2rad([lo,hi])))[0]*(180/np.pi)**2
    result=dict(status='metadata_frozen_before_new_motion_results',nominal_area_sqdeg=float(area),frozen_science_cores=len(plans['fits']),object_route_estimate=counts,
        all_to_legacy_reported_measurement_ratio=counts['all_objects']['reported_measurements']/counts['legacy_r_index']['reported_measurements'],
        planning_raw_gib_at_300_bytes_per_reported_measurement=counts['all_objects']['reported_measurements']*300/1024**3,
        conservative_chip_opportunities=len(selected),by_instrument_band=bands,original_reference_sha256=digest(reference),
        source_plan_sha256=digest(state/'sources.json'),fit_plan_sha256=digest(state/'fits.json'),geometry_sha256=digest(geometry),slurm_job=os.environ['SLURM_JOB_ID'],
        scope='Whole predeclared C rectangle and original half-open core plan; metadata and reported object ndet before reading any new motion result. No r-object cut in all_objects estimate. SUM(ndet) includes object measurements outside the core and may differ from actual published meas; this is a resource estimate, not scientific detection count. Chip bounding-box opportunities are conservative, not certified available area or target SNR. Original C0 support is frozen for planning; additional groups need independent galaxy calibration.')
    check_aggregate(result);save_json(work/'nsc_continuous_preflight_summary.json',result)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('work','state','geometry'):p.add_argument(n,type=Path)
    a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use a Slurm compute node')
    a.work.mkdir(parents=True,exist_ok=True);bind(a.work,'old_reference_receipt.json',verify_directory(a.state/'shared',('private_c0.npz',)));bind(a.work,'geometry.toml',a.geometry)
    for name in ('fits','sources','references'):bind(a.work,'frozen_'+name+'.json',a.state/(name+'.json'))
    with run_stage(a.work,a.geometry,'whole_c','nsc-continuous-preflight-v15',inputs=('old_reference_receipt.json','geometry.toml','frozen_fits.json','frozen_sources.json','frozen_references.json'),outputs=('nsc_continuous_preflight_summary.json','private_continuous_chip_opportunities.csv.gz','private_frozen_fits.json','private_frozen_sources.json','private_frozen_references.json')) as go:
        if go:run(a.work,a.state,a.geometry)


if __name__=='__main__':main()
