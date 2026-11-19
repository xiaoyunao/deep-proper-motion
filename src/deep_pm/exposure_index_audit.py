"""Bounded exposure/CCD audit of the historical object-index selection.

A fixed central 0.05 by 0.05 degree cell, at most two physical exposures per
actual filter, count-checked queries and explicit metadata coverage. These are
retrieval comparisons; NSC object IDs never become association truth.
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.path import Path as Polygon

from . import access,normalize
from .__main__ import tomllib
from .allband_qso import physical_candidates
from .astrometry import project
from .continuous_reference import PRECISION,bind,read_offsets,verify_directory
from .io import save_json
from .precision_v03 import calibration_groups,calibration_support
from .provenance import run_stage
from .public_summary import check_aggregate
from .regional_preflight import CHIP_VERTICES


def run(work,source,shared,spec):
    center=tuple(float(np.mean(spec[k])) for k in ('ra','dec'));cache=access.Cache(work/'cache')
    columns=['exposure','ccdnum','instrument','filter','mjd']+CHIP_VERTICES
    # Only tiny metadata cone; no scan of meas for a broad spatial region.
    chips=access.checked_spatial(cache,'nsc',','.join(columns),'nsc_dr2.chip',access.q3c(*center,.40),center[1]-.400001,center[1]+.400001)
    covering=[]
    for j,row in chips.iterrows():
        corners=np.array([(row[f'vertex_ra{k}'],row[f'vertex_dec{k}']) for k in range(1,5)])
        if np.isfinite(corners).all() and Polygon(project(*corners.T,center)).contains_point([0.,0.]):covering.append(j)
    candidates=chips.loc[covering].drop_duplicates(['exposure','ccdnum']).sort_values(['filter','mjd','exposure'])
    selected=[]
    for _,part in candidates.groupby('filter',sort=True):
        # First and last opportunity; do not choose by astrometric result or magnitude.
        visits=part.drop_duplicates('exposure');selected.extend(visits.index[[0,-1]] if len(visits)>1 else visits.index)
    chips=candidates.loc[sorted(set(selected))];chips.to_csv(work/'private_selected_chips.csv',index=False)
    parts=[]
    for row in chips.itertuples():
        where=f'exposure={access.literal(str(row.exposure))} AND ccdnum={int(row.ccdnum)} AND ('+access.bbox(*center,.025)+')'
        # COUNT plus exact returned rows; never silently accept a LIMIT-truncated scientific sample.
        parts.append(access.checked_spatial(cache,'nsc',access.MEAS,'nsc_dr2.meas',where,center[1]-.0250001,center[1]+.0250001))
    measurements=pd.concat(parts,ignore_index=True) if parts else pd.DataFrame(columns=access.MEAS.split(','))
    if measurements.measid.duplicated().any():raise ValueError('Duplicated physical measurements in disjoint exposure/CCD audit')
    measurements.to_csv(work/'private_exposure_measurements.csv.gz',index=False)
    original=pd.read_csv(source/'detections.csv',usecols=['detection_id'],dtype=str)
    old_index=pd.read_csv(source/'official_nsc.csv',dtype=str);old_ids=set(old_index['id'])
    present=('NSC:'+measurements.measid.astype(str)).isin(original.detection_id).to_numpy();missing=measurements.loc[~present]
    objects=[];ids=sorted(set(missing.objectid.astype(str)))
    for start in range(0,len(ids),250):
        where='id IN ('+','.join(map(access.literal,ids[start:start+250]))+')'
        objects.append(cache.checked('nsc','id AS objectid,ndetr,nphotr,rmag','nsc_dr2.object',where))
    metadata=pd.concat(objects,ignore_index=True) if objects else pd.DataFrame(columns=['objectid','ndetr','nphotr','rmag'])
    metadata.to_csv(work/'private_missing_object_metadata.csv',index=False)
    reasons=dict(no_r_detection_objects=int((pd.to_numeric(metadata.ndetr)<=0).sum()),
        r_photometry_fainter_than_index_cut_objects=int(((pd.to_numeric(metadata.rmag)>=24)&(pd.to_numeric(metadata.nphotr)>0)).sum()),
        missing_measurements_from_original_index_objects=int(missing.objectid.astype(str).isin(old_ids).sum()))
    normalized=normalize.nsc(measurements);valid=physical_candidates(normalized) if len(normalized) else np.zeros(0,bool)
    support=calibration_support(calibration_groups(normalized),read_offsets(shared/'private_c0.npz')) if len(normalized) else np.zeros(0,bool)
    byband={}
    for band,part in measurements.groupby('filter',sort=True):
        ix=part.index.to_numpy();byband[str(band)]=dict(returned_measurements=len(ix),present_in_cached_input=int(present[ix].sum()),missing_from_cached_input=int((~present[ix]).sum()),
            physical_quality_candidates=int(valid[ix].sum()),missing_physical_candidates=int((~present[ix]&valid[ix]).sum()),
            missing_physical_reference_supported=int((~present[ix]&valid[ix]&support[ix]).sum()))
    result=dict(status='completed' if len(chips) else 'no_chips_cover_fixed_center',selected_physical_exposures=chips.exposure.nunique(),
        metadata_covering_chips=len(candidates),by_filter=byband,missing_object_filter_diagnostics=reasons,
        query_cell_side_dec_deg=.05,scope='Fixed geometric core center, first/last actual exposure per filter. Comparison against retrieved normalized measurement identities, not source truth or image completeness. Historical ndetr/r selection can suppress non-r discoveries; zero audit losses in a few exposures do not prove general completeness.',
        slurm_job=os.environ['SLURM_JOB_ID'])
    check_aggregate(result);save_json(work/'index_route_audit_summary.json',result)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('work','source','shared','geometry'):p.add_argument(name,type=Path)
    p.add_argument('region');a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use a Slurm compute node')
    a.work.mkdir(parents=True,exist_ok=True);bind(a.work,'source_receipt.json',verify_directory(a.source,('detections.csv','official_nsc.csv')))
    bind(a.work,'reference_receipt.json',verify_directory(a.shared,('private_c0.npz',)));bind(a.work,'geometry.toml',a.geometry)
    with run_stage(a.work,PRECISION,a.region,'exposure-index-audit-v14',inputs=('source_receipt.json','reference_receipt.json','geometry.toml'),outputs=('index_route_audit_summary.json','private_selected_chips.csv','private_exposure_measurements.csv.gz','private_missing_object_metadata.csv')) as go:
        if go:run(a.work,a.source,a.shared,tomllib.loads(a.geometry.read_text())['fields'][a.region])


if __name__=='__main__':main()
