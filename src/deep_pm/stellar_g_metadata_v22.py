"""At most two existing cores chosen by g geometry alone, before new residuals."""
import argparse
from collections import defaultdict
from functools import lru_cache
from itertools import combinations
import os
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.time import Time
from matplotlib.path import Path as Polygon

from .astrometry import project
from .blind_catalogue_quality import read
from .continuous_instruments import chip_receipts
from .dcr_geometry import zenith_vectors,tangent_geometry
from .io import save_json,digest
from .public_summary import check_aggregate


@lru_cache(maxsize=100000)
def cached_zenith(mjd):return zenith_vectors(np.array([mjd]))


def run(root,output):
    output.mkdir(parents=True,exist_ok=True)
    plan_path=root/'v2.2/registration/private_full_coordinate_plan.json';plan=read(plan_path)
    evaluations=[];skipped=defaultdict(int)
    for entry in plan['cores']:
        source=Path(entry['source']);paths,complete=chip_receipts(source/'run_manifest.json')
        if not complete:skipped['incomplete_chip_metadata']+=1;continue
        parts=[]
        for path in paths:
            f=pd.read_csv(path,dtype={'exposure':str})
            if not {'filter','instrument','mjd','airmass','ccdnum','vertex_ra1','vertex_dec1'}<=set(f):continue
            parts.append(f.loc[f.instrument.eq('c4d')&f['filter'].eq('g')])
        if not parts:skipped['no_g_chip_metadata']+=1;continue
        frame=pd.concat(parts,ignore_index=True).drop_duplicates(['exposure','ccdnum'])
        if frame.empty:continue
        spec=entry['spec'];center=[sum(spec['ra'])/2,sum(spec['dec'])/2]
        # Footprints, not detected stars or resulting proper motions.
        ra=np.linspace(*spec['ra'],7)[1:-1];dec=np.linspace(*spec['dec'],7)[1:-1];sky=np.array(np.meshgrid(ra,dec)).reshape(2,-1).T;xy=project(*sky.T,center)
        visits=[]
        for name,part in frame.groupby('exposure',sort=True):
            covered=np.zeros(len(xy),bool)
            for chip in part.itertuples():
                vertices=np.array([[getattr(chip,'vertex_ra'+str(k)),getattr(chip,'vertex_dec'+str(k))] for k in range(1,5)])
                if not np.isfinite(vertices).all():continue
                points=project(*vertices.T,center);covered|=Polygon(np.vstack([points,points[0]])).contains_points(xy)
            if not covered.any():continue
            mjd=float(part.mjd.median())
            if not np.isfinite(mjd):skipped['missing_exposure_mjd']+=1;continue
            stamp=Time([mjd],format='mjd',scale='utc')
            try:vector,air=tangent_geometry(np.array([center[0]]),np.array([center[1]]),cached_zenith(mjd),center)
            except ValueError:skipped['outside_geometric_altitude_support']+=1;continue
            visits.append(dict(exposure=name,night=stamp.isot[0][:10],mjd_utc=mjd,metadata_airmass=float(part.airmass.median()),
                geometric_airmass=float(air[0]),h=vector[0],covered=covered))
        repeat=[]
        for left,right in combinations(visits,2):
            if left['night']!=right['night']:continue
            overlap=int((left['covered']&right['covered']).sum())
            if not overlap:continue
            span=float(np.linalg.norm(left['h']-right['h']))
            repeat.append(dict(night=left['night'],overlap_grid_fraction=overlap/len(sky),vector_span=span,
                time_separation_hours=abs(left['mjd_utc']-right['mjd_utc'])*24))
        useful=[r for r in repeat if r['vector_span']>=.1 and r['overlap_grid_fraction']>=.2]
        item=dict(core=entry['core'],window=entry['window'],g_exposures=len(visits),
            median_airmass=float(np.median([r['geometric_airmass'] for r in visits])) if visits else None,
            usable_geometry_nights=len({r['night'] for r in useful}),maximum_vector_span=max([r['vector_span'] for r in repeat],default=0.),
            overlapping_repeat_pairs=len(repeat))
        evaluations.append(item)
    # Two different scientific roles; no PM, residual or validation quality read.
    low=[r for r in evaluations if r['g_exposures']>=3 and r['median_airmass'] is not None and r['median_airmass']<=1.4]
    low=sorted(low,key=lambda r:(-r['usable_geometry_nights'],-r['maximum_vector_span'],-r['g_exposures'],r['core']))
    repeat=sorted(evaluations,key=lambda r:(-r['usable_geometry_nights'],-r['maximum_vector_span'],-r['g_exposures'],r['core']))
    chosen=[]
    if low:chosen.append(dict(low[0],role='low_airmass_DECam'))
    best=next((r for r in repeat if r['core'] not in {c['core'] for c in chosen} and r['overlapping_repeat_pairs']>0),None)
    if best is not None:chosen.append(dict(best,role='repeat_geometry_DECam'))
    selected=[dict(next(e for e in plan['cores'] if e['core']==c['core']),metadata_selection=c) for c in chosen]
    save_json(output/'private_additional_core_plan.json',dict(source_plan_sha256=digest(plan_path),selected=selected,maximum_additional_cores=2,
        criteria='5x5 real CCD footprint overlap >=0.2; geometric tanZ-vector span>=0.1 within UTC night; low-airmassrole mediansecz<=1.4. Rank by independent useful nights, vector span, then exposure count. No new residual/PM read.'))
    result=dict(source_plan_sha256=digest(plan_path),evaluated_cores=len(evaluations),skipped=dict(skipped),
        low_airmass_candidates=len(low),at_least_three_useful_repeat_nights=sum(r['usable_geometry_nights']>=3 for r in evaluations),
        chosen=chosen,private_selection_sha256=digest(output/'private_additional_core_plan.json'),slurm_job=os.environ['SLURM_JOB_ID'],
        scope='Metadata-only existing-coverage selection, no new catalogue outcome. Actual CCD25point overlap proxy, not effective-pixel or star support; next source-level support must verify. Other sites not assigned Blanco; unsupported altitudes recorded by task failure rather than fabricated geometry.')
    check_aggregate(result);save_json(output/'v22_additional_g_core_selection_summary.json',result)
    print('ADDITIONAL_G_METADATA_SELECTED',chosen,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use Slurm')
    run(**vars(a))
