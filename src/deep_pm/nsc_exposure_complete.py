"""All-exposure/CCD retrieval in a predeclared tiny cell, without an r index.

Completeness is conditional on the returned official chip metadata and tested
search margin; it is not image extraction completeness or a blind source label.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import os
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.path import Path as Polygon

from . import access,normalize
from .__main__ import tomllib
from .allband_qso import physical_candidates
from .astrometry import project,unproject
from .continuous_reference import PRECISION,bind,read_offsets,verify_directory
from .io import save_json
from .precision_v03 import calibration_groups,calibration_support
from .provenance import run_stage
from .public_summary import check_aggregate
from .regional_preflight import CHIP_VERTICES
from .nsc_information import night_labels


def intersecting(chips,center,half):
    box=np.array([[-half,-half],[half,-half],[half,half],[-half,half],[-half,-half]])*3600000
    square=Polygon(box,closed=True);selected=[]
    for j,row in chips.iterrows():
        corners=np.array([(row[f'vertex_ra{k}'],row[f'vertex_dec{k}']) for k in range(1,5)])
        if not np.isfinite(corners).all():continue
        points=project(*corners.T,center);poly=Polygon(np.vstack([points,points[0]]),closed=True)
        if poly.intersects_path(square,filled=True):selected.append(j)
    return chips.loc[selected].drop_duplicates(['exposure','ccdnum']).reset_index(drop=True)


def run(work,source,shared,spec,half,workers,center_band=None):
    center=tuple(float(np.mean(spec[k])) for k in ('ra','dec'));cache=access.Cache(work/'cache')
    if center_band is not None:
        from .exposure_catalogues import inside
        parent=pd.read_csv(source/'detections.csv',usecols=['survey','band','ra_deg','dec_deg'],low_memory=False)
        selected=parent.loc[parent.survey.eq('NSC')&parent.band.eq(center_band)&inside(parent.ra_deg,parent.dec_deg,spec)]
        if not len(selected):raise ValueError('Requested band has no retrieved positions in the declared parent core')
        # Geometry only: neither motion, residual, brightness nor error chooses
        # the targeted band footprint. All bands are fetched at the new center.
        center=tuple(unproject(np.median(project(selected.ra_deg,selected.dec_deg,center),axis=0),center)[0])
    columns=['exposure','ccdnum','ra','dec','instrument','airmass','filter','mjd']+CHIP_VERTICES
    surveys=[];metadata=[]
    for margin in (.60,1.20):
        radius=float(np.sqrt(2)*half+margin)
        table=access.checked_spatial(cache,'nsc',','.join(columns),'nsc_dr2.chip',access.q3c(*center,radius),center[1]-radius-1e-7,center[1]+radius+1e-7)
        selected=intersecting(table,center,half);metadata.append(selected);surveys.append(set(map(tuple,selected[['exposure','ccdnum']].to_numpy())))
    if surveys[0]!=surveys[1]:raise ValueError('Chip coverage changes with metadata margin; do not declare the tiny cell complete')
    chips=metadata[-1].sort_values(['mjd','exposure','ccdnum']).reset_index(drop=True)
    if chips.duplicated(['exposure','ccdnum']).any():raise ValueError('Physical chip metadata not unique')
    chips.to_csv(work/'private_all_covering_chips.csv',index=False)
    def fetch(item):
        exposure,part=item;ccds=','.join(str(int(v)) for v in sorted(set(part.ccdnum)))
        where=f'exposure={access.literal(str(exposure))} AND ccdnum IN ({ccds}) AND ('+access.bbox(*center,half*1.01)+')'
        return access.checked_spatial(cache,'nsc',access.MEAS,'nsc_dr2.meas',where,center[1]-half*1.01,center[1]+half*1.01)
    frames=[]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for number,frame in enumerate(pool.map(fetch,chips.groupby('exposure',sort=True)),1):
            frames.append(frame)
            if number%20==0:print('Count-verified physical exposures',number,'/',chips.exposure.nunique(),flush=True)
    table=pd.concat(frames,ignore_index=True) if frames else pd.DataFrame(columns=access.MEAS.split(','))
    if len(table):table=table.loc[(abs(project(table.ra,table.dec,center)/3600000)<=half).all(axis=1)].reset_index(drop=True)
    if table.measid.duplicated().any():raise ValueError('Duplicate physical detections in disjoint exposures')
    table=table.merge(chips[['exposure','ccdnum','instrument','airmass']].drop_duplicates(),on=['exposure','ccdnum'],how='left',validate='many_to_one')
    if table.instrument.isna().any():raise ValueError('Measurement lacks physical chip metadata')
    table.to_csv(work/'private_exposure_measurements.csv.gz',index=False)
    data=normalize.nsc(table);data['night_utc18']=night_labels(data) if len(data) else pd.Series(dtype=int);data['epoch_tt_year']=2000+(data.observation_time-51544.5)/365.25
    solution=read_offsets(shared/'private_c0.npz');valid=physical_candidates(data) if len(data) else np.zeros(0,bool);support=calibration_support(calibration_groups(data),solution) if len(data) else np.zeros(0,bool)
    data['reference_supported']=support;data['physical_candidate']=valid;data.to_csv(work/'private_exposure_detections.csv.gz',index=False)
    old=pd.read_csv(source/'detections.csv',usecols=['detection_id','survey','ra_deg','dec_deg'],low_memory=False)
    present=data.detection_id.isin(old.detection_id).to_numpy();returned=set(data.detection_id)
    # Compare the exact queried physical exposures/CCDs and geometry, rather than
    # all old object-index measurements outside the tested opportunity universe.
    selected=set('NSC:'+chips.exposure.astype(str));raw=pd.read_csv(source/'detections.csv',usecols=['detection_id','survey','exposure_id','detector_id','ra_deg','dec_deg'],low_memory=False)
    xy=project(raw.ra_deg,raw.dec_deg,center)/3600000;inside=(abs(xy)<=half).all(axis=1)&raw.survey.eq('NSC').to_numpy()
    known_chip=set((str(row.exposure),int(row.ccdnum)) for row in chips.itertuples())
    physical=np.array([(str(e).split(':',1)[-1],int(c)) in known_chip for e,c in zip(raw.exposure_id,raw.detector_id)])
    expected=raw.loc[inside&physical,'detection_id'];missing_return=sorted(set(expected)-returned)
    if missing_return:raise ValueError('Exposure query missed previously retrieved detections in the audited footprint')
    qualified_r_objects=set(data.loc[valid&data.band.eq('r').to_numpy(),'original_object_id']);labels=data.original_object_id
    byband={}
    for band,part in data.groupby('band',sort=True):
        i=part.index.to_numpy();good=valid[i]&support[i]
        byband[str(band)]=dict(measurements=len(i),exposures=part.exposure_id.nunique(),instruments=dict(Counter(part.instrument)),
            nights=part.night_utc18.nunique(),physical_candidates=int(valid[i].sum()),reference_supported_candidates=int(good.sum()),
            not_in_old_object_route=int((~present[i]).sum()),missing_physical_supported=int((~present[i]&good).sum()),
            object_index_labels_without_qualified_r=int(labels.iloc[i[good]].loc[~labels.iloc[i[good]].isin(qualified_r_objects)].nunique()))
    result=dict(status='completed',selected_physical_exposures=chips.exposure.nunique(),covering_chips=len(chips),measurement_count=len(data),by_filter=byband,
        old_input_intersection_count=int(present.sum()),previously_returned_measurements_checked=len(expected),previously_returned_missing_count=len(missing_return),
        metadata_margin_stability=True,metadata_search_margins_deg=[.60,1.20],cell_side_tangent_deg=2*half,center_selection='parent_core_center' if center_band is None else 'retrieved_'+center_band+'_position_footprint_no_motion_or_quality_selection',slurm_job=os.environ['SLURM_JOB_ID'],
        scope='All returned official CCD footprints intersecting one predeclared tangent square, every physical exposure queried with independent COUNT and row checks. Two metadata search margins give identical intersecting chips. Conditional catalogue-measurement coverage, not proof of unknown chips outside the search or image extraction completeness. Object IDs describe retrieval labels only, never new-association truth. Every actual band is retained; no TOP/sample exposure cap.')
    check_aggregate(result);save_json(work/'nsc_exposure_complete_summary.json',result)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('work','source','shared','geometry'):p.add_argument(n,type=Path)
    p.add_argument('region');p.add_argument('--center-band');p.add_argument('--half-width',type=float,default=.025);p.add_argument('--workers',type=int,default=2);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use a Slurm compute node')
    if not 0<a.half_width<=.05:p.error('This is a tiny-cell validation, not broad production')
    a.work.mkdir(parents=True,exist_ok=True);bind(a.work,'source_receipt.json',verify_directory(a.source,('detections.csv',)));bind(a.work,'reference_receipt.json',verify_directory(a.shared,('private_c0.npz',)));bind(a.work,'geometry.toml',a.geometry)
    settings=dict(half_width=a.half_width,workers=a.workers,center_band=a.center_band);options=a.work/'options.json'
    if options.exists() and __import__('json').loads(options.read_text())!=settings:raise ValueError('Changed exposure audit options')
    if not options.exists():save_json(options,settings)
    with run_stage(a.work,PRECISION,a.region,'nsc-exposure-complete-v15',inputs=('source_receipt.json','reference_receipt.json','geometry.toml','options.json'),outputs=('nsc_exposure_complete_summary.json','private_all_covering_chips.csv','private_exposure_measurements.csv.gz','private_exposure_detections.csv.gz')) as go:
        if go:run(a.work,a.source,a.shared,tomllib.loads(a.geometry.read_text())['fields'][a.region],a.half_width,a.workers,a.center_band)


if __name__=='__main__':main()
