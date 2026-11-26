"""Frozen-coefficient validation along zenith and deep-neighbour directions.

No coefficient training; Gaia residuals are independent bright-end readouts.
Binned trends remain conditional descriptions with repeated measurements.
"""
import argparse
from pathlib import Path
import os
import numpy as np
from .astrometry import basis
from .blind_catalogue_quality import read
from .io import digest,save_json
from .public_summary import check_aggregate
from .stellar_g_mechanism_v22 import COEFFICIENTS,finite_json


def summary(rows):
    if len(rows)<5:return dict(measurements=len(rows))
    residual=np.array([r['native_residual_mas'] for r in rows]);h=np.array([r['native_geometry'] for r in rows]);color=np.array([r['color'] for r in rows]);unit=h/np.linalg.norm(h,axis=1)[:,None]
    result=dict(measurements=len(rows),physical_stars=len({tuple(r['source_key']) for r in rows}),
        color_p16_p50_p84=np.quantile(color,[.16,.5,.84]),actual_g_SNR_p16_p50_p84=np.quantile([r['band_snr'] for r in rows],[.16,.5,.84]))
    models={}
    for k in (0.,)+COEFFICIENTS:
        r=residual-k*(color-1.2)[:,None]*h
        model=dict(zenith_parallel_median_mas=float(np.median(np.sum(r*unit,axis=1))),zenith_perpendicular_median_mas=float(np.median(r[:,1]*unit[:,0]-r[:,0]*unit[:,1])),absolute_radial_P68_mas=float(np.quantile(np.linalg.norm(r,axis=1),.68)))
        selected=[i for i,v in enumerate(rows) if 0<v['deep_neighbour'].get('neighbour_distance_mas',10000)<10000]
        if len(selected)>=5:
            direction=np.array([[rows[i]['deep_neighbour']['neighbour_east_mas'],rows[i]['deep_neighbour']['neighbour_north_mas']] for i in selected]);direction/=np.linalg.norm(direction,axis=1)[:,None]
            model.update(deep_neighbor_measurements=len(selected),neighbor_parallel_median_mas=float(np.median(np.sum(r[selected]*direction,axis=1))),neighbor_perpendicular_median_mas=float(np.median(r[selected,1]*direction[:,0]-r[selected,0]*direction[:,1])))
        models[str(k)]=model
    result['frozen_models']=models;return result


def run(root,output):
    source=root/'v2.2/stellar-g/private_stellar_g_mechanism.json';rows=read(source)['measurements'];groups={}
    for instrument in sorted({r['instrument'] for r in rows}):
        part=[r for r in rows if r['instrument']==instrument]
        for lo,hi in [(20,40),(40,100),(100,100000)]:
            supported=[r for r in part if lo<=r['band_snr']<hi]
            for clo,chi in [(-1,.5),(.5,1),(1,1.5),(1.5,2),(2,4)]:
                color=[r for r in supported if clo<=r['color']<chi]
                for tag,condition in [('all',lambda r:True),('neighbor_lt3arcsec',lambda r:r['deep_neighbour'].get('neighbour_distance_mas',10000)<3000),('neighbor_ge3arcsec',lambda r:r['deep_neighbour'].get('neighbour_distance_mas',0)>=3000),('seeing_lt1p2',lambda r:r.get('seeing') is not None and r['seeing']<1.2),('seeing_ge1p2',lambda r:r.get('seeing') is not None and r['seeing']>=1.2)]:
                    groups[f'{instrument}__gSNR{lo}_{hi}__color{clo}_{chi}__{tag}']=summary([r for r in color if condition(r)])
    result=dict(input_sha256=digest(source),groups=groups,slurm_job=os.environ['SLURM_JOB_ID'],
        scope='Fixed historical coefficients only; bright Gaia validation never used to train the new repeated-star coefficient. Same actual g SNR and colour intervals; neighbor direction from deep LS within10arcsec. Seeing is native published arcsec. Conditional measurement medians, no independent-object/night intervals or faint-star certification. Sparse neighbor strata cannot establish blending causality; partially shared LS images are not external stellar truth.')
    result=finite_json(result);check_aggregate(result);output.mkdir(exist_ok=True);save_json(output/'v22_stellar_g_neighbour_direction_summary.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use Slurm')
    run(a.root,a.output)
