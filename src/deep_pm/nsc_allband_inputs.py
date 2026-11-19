"""Remove the historical r-object acquisition cut while preserving SDSS r data.

The all-object join is checked against exposure/CCD truth only as a retrieval
universe; object IDs never define correct blind associations. Ordinary-motion
halo and object-centroid retrieval limits remain explicit.
"""
import argparse
import json
from collections import Counter
import json
import os
from pathlib import Path
import resource
from time import monotonic

import numpy as np
import pandas as pd

from . import access,normalize
from .__main__ import tomllib
from .continuous_reference import FILES,bind,verify_directory
from .exposure_catalogues import inside
from .exposure_inputs import tap_join,tap_rectangle
from .exposure_preflight import where_box
from .io import save_json,digest
from .provenance import run_stage
from .public_summary import check_aggregate


def acquire(work,parent,spec,config,workers,exposure_audit=None,reuse_r_archive=False):
    start=monotonic();cache=access.Cache(work/'cache');halo=config['halo_deg'];lo,hi=spec['dec'];where=where_box(spec,halo)
    cols='id AS objectid,ra,dec,pmra,pmdec,pmraerr,pmdecerr,deltamjd,ndet'
    archive_where=where
    if reuse_r_archive:
        parent_summary=json.loads((parent/'acquisition_summary.json').read_text())
        if (parent_summary.get('measurement_route')!='datalab_native_archive_join'
                or not parent_summary.get('native_count_window')
                or parent_summary.get('all_nsc_object_selection')):
            raise ValueError('The previous cache must use the signed native r-selected archive join')
        parent_manifest=json.loads((parent/'run_manifest.json').read_text())
        snapshot=parent/'config_snapshot.toml'
        original=tomllib.loads(snapshot.read_text())
        if (digest(snapshot)!=parent_manifest['config_sha256']
                or original['fields'][parent_manifest['region']]!=spec or original['halo_deg']!=halo):
            raise ValueError('The r-selected archive complement needs the same signed footprint')
        archive_where+=" AND ((ndetr>0 AND (rmag<24 OR nphotr=0 OR rmag IS NULL)) IS NOT TRUE)"
    objects=access.checked_spatial(cache,'nsc',cols,'nsc_dr2.object',archive_where,lo-halo-1e-7,hi+halo+1e-7)
    if objects.objectid.duplicated().any():raise ValueError('Duplicate all-band acquisition object aliases')
    print('All-filter retrieval objects',len(objects),flush=True)
    nsc=tap_join(cache,objects,spec,halo,service='nsc',workers=workers,native_cap=config.get('native_row_cap',80000),
                 native_window=True,all_objects=not reuse_r_archive,complement_r=reuse_r_archive)
    if reuse_r_archive:
        frozen_objects=pd.read_csv(parent/'official_nsc.csv').rename(columns={'id':'objectid'})
        if set(frozen_objects.objectid.astype(str)).intersection(objects.objectid.astype(str)):
            raise ValueError('Frozen r archive and current complement overlap')
        old_nsc=pd.read_csv(parent/'nsc_object.csv',low_memory=False)
        if not set(old_nsc.objectid.astype(str)) <= set(frozen_objects.objectid.astype(str)):
            raise ValueError('The frozen archive contains unexpected object aliases')
        nsc=pd.concat([old_nsc,nsc],ignore_index=True)
        objects=pd.concat([frozen_objects,objects],ignore_index=True)
    objects.rename(columns={'objectid':'id'}).to_csv(work/'official_nsc.csv',index=False)
    if not set(nsc.objectid.astype(str))<=set(objects.objectid.astype(str)):raise ValueError('Unexpected retrieval object alias')
    nsc=nsc.loc[inside(nsc.ra,nsc.dec,spec,halo)].reset_index(drop=True)
    if nsc.measid.duplicated().any():raise ValueError('Frozen archive and complement have duplicate measurements')
    nsc.to_csv(work/'nsc_object.csv',index=False)
    pad=halo+.65;chips=access.checked_spatial(cache,'nsc','exposure,ccdnum,instrument,airmass','nsc_dr2.chip',where_box(spec,pad),lo-pad-1e-7,hi+pad+1e-7)
    chips=chips.drop_duplicates()
    if chips.duplicated(['exposure','ccdnum']).any():raise ValueError('Conflicting physical chip metadata')
    chips.to_csv(work/'chips.csv',index=False)
    raw=nsc.merge(chips,on=['exposure','ccdnum'],how='left',validate='many_to_one');late=normalize.nsc(raw)
    old=pd.read_csv(parent/'detections.csv',low_memory=False);early=old.loc[old.survey.eq('SDSS')].copy()
    if not early.band.eq('r').all():raise ValueError('The frozen SDSS astrometric input must be r only')
    labels=pd.read_csv(parent/'external_labels.csv',dtype=str).set_index('detection_id').original_object_id
    early_labels=early.detection_id.map(labels)
    label_table=pd.concat([pd.DataFrame(dict(detection_id=early.detection_id,original_object_id=early_labels)),late[['detection_id','original_object_id']]],ignore_index=True)
    data=pd.concat([early,late.drop(columns='original_object_id')],ignore_index=True)
    if data.detection_id.duplicated().any():raise ValueError('Duplicate physical detection in all-band normalized input')
    data=data.sort_values(['observation_time','detection_id']).reset_index(drop=True);data.to_csv(work/'detections.csv',index=False);label_table.to_csv(work/'external_labels.csv',index=False)
    for relative in ('sdss.csv','spectra_union.csv','spectra_veto.csv','validation/gaia.csv'):
        target=work/relative;target.parent.mkdir(parents=True,exist_ok=True);origin=(parent/relative).resolve()
        if not origin.exists():
            if relative=='sdss.csv':continue
            raise ValueError('Missing frozen ancillary input '+relative)
        if target.is_symlink() and target.resolve()!=origin:raise ValueError('Frozen ancillary alias changed')
        if not target.exists():target.symlink_to(origin)
    expected=old.loc[old.survey.eq('NSC')&inside(old.ra_deg,old.dec_deg,spec,halo)].copy();actual=late.set_index('detection_id');missing=set(expected.detection_id)-set(actual.index)
    if missing:raise ValueError('Unfiltered object join missed old published detections inside the same halo')
    compare=['ra_deg','dec_deg','observation_time','sigma_east_mas','sigma_north_mas','magnitude','magnitude_error']
    for key in compare:np.testing.assert_allclose(actual.loc[expected.detection_id,key],expected[key],rtol=1e-12,atol=1e-10,equal_nan=True)
    sdss_lookup=data.loc[data.survey.eq('SDSS')].set_index('detection_id').reindex(early.detection_id)
    for key in compare:np.testing.assert_allclose(sdss_lookup[key],early[key],rtol=0,atol=0,equal_nan=True)
    audit=None
    if exposure_audit is not None:
        verify_directory(exposure_audit,('private_exposure_detections.csv.gz','nsc_exposure_complete_summary.json'));truth=pd.read_csv(exposure_audit/'private_exposure_detections.csv.gz',usecols=['detection_id'])
        absent=set(truth.detection_id)-set(actual.index)
        if absent:raise ValueError('All-object retrieval failed the independent complete exposure-cell comparison')
        audit=dict(exposure_measurements=len(truth),recovered=len(truth),missing=0,audit_receipt_sha256=digest(exposure_audit/'run_manifest.json'))
    result=dict(retrieval_objects=len(objects),downloaded_nsc_measures=len(nsc),old_nsc_measures_in_halo=len(expected),new_nsc_measures=len(late)-len(expected),
        normalized_detections=len(data),sdss_r_measurements_preserved=len(early),raw_sdss_photometry_available=(parent/'sdss.csv').is_file(),by_nsc_band=dict(Counter(late.band)),chip_metadata_entries=len(chips),
        retrieved_using_signed_r_archive_complement=reuse_r_archive,exposure_cell_comparison=audit,elapsed_seconds=monotonic()-start,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,slurm_job=os.environ['SLURM_JOB_ID'],
        scope='All NSC object-centroid aliases in the declared core+halo, no ndetr/rmag/nphotr/filter cut; count-checked meas join and physical-ID dedup. Original SDSS r detections preserved exactly including every physical run. Object labels go only to separate auxiliary labels, not association input. Source identities still require blind association. Halo/object-centroid selection and published-measurement extraction limits remain; actual exposure-cell comparison is separate from image completeness.')
    check_aggregate(result);save_json(work/'acquisition_summary.json',result)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('work','parent','geometry'):p.add_argument(n,type=Path)
    p.add_argument('region');p.add_argument('--workers',type=int,default=1);p.add_argument('--exposure-audit',type=Path);p.add_argument('--reuse-r-archive',action='store_true');a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use a Slurm compute node')
    # Early v0.9 receipts signed normalized SDSS positions, not the ancillary
    # raw photometry file. Preserve those signed positions and bind today's raw
    # file separately; do not imply a historical checksum that never existed.
    a.work.mkdir(parents=True,exist_ok=True)
    bind(a.work,'parent_receipt.json',verify_directory(a.parent,FILES+(('nsc_object.csv',) if a.reuse_r_archive else ())))
    bind(a.work,'geometry.toml',a.geometry)
    inputs=['parent_receipt.json','geometry.toml']
    if (a.parent/'sdss.csv').is_file():
        bind(a.work,'parent_sdss.csv',a.parent/'sdss.csv');inputs.append('parent_sdss.csv')
    if a.exposure_audit:bind(a.work,'exposure_audit_receipt.json',verify_directory(a.exposure_audit,('private_exposure_detections.csv.gz','nsc_exposure_complete_summary.json')));inputs.append('exposure_audit_receipt.json')
    expected=dict(all_nsc_objects=True,sdss_positions='frozen_r_only',workers=a.workers)
    if a.reuse_r_archive:
        bind(a.work,'parent_geometry.toml',a.parent/'config_snapshot.toml')
        inputs.append('parent_geometry.toml')
        expected['reuse_signed_r_archive']=True
    options=a.work/'options.json'
    if options.exists() and json.loads(options.read_text())!=expected:raise ValueError('Changed all-object acquisition options')
    if not options.exists():save_json(options,expected)
    inputs.append('options.json');config=tomllib.loads(a.geometry.read_text())
    outputs=FILES+('sdss.csv','nsc_object.csv','chips.csv')
    with run_stage(a.work,a.geometry,a.region,'nsc-allband-inputs-v15',inputs=inputs,outputs=outputs) as go:
        if go:acquire(a.work,a.parent,config['fields'][a.region],config,a.workers,a.exposure_audit,a.reuse_r_archive)


if __name__=='__main__':main()
