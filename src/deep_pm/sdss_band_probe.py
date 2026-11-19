"""Inspect actual DR17 single-scan centroid, time and calibration arrays."""
import argparse
import json
import os
from pathlib import Path

from astropy.io import fits
import numpy as np
import pandas as pd
import requests

from .astrometry import sdss_tai_to_tt
from .continuous_reference import bind,verify_directory
from .io import digest,save_json
from .provenance import run_stage
from .public_summary import check_aggregate

BANDS='ugriz'


def arrays(table):
    return {name.upper():name for name in table.names}


def schema(hdu):
    return {c.name:dict(format=c.format,unit=c.unit) for c in hdu.columns}


def file_at(work,sas,relative):
    local=sas/relative
    if local.is_file():return local
    target=work/'cache'/relative;target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists():
        url='https://data.sdss.org/sas/dr17/eboss/'+str(relative)
        response=requests.get(url,timeout=(12,100));response.raise_for_status()
        if not response.content.startswith(b'SIMPLE'):raise ValueError('Expected a real FITS file')
        temp=target.with_suffix('.part');temp.write_bytes(response.content);temp.replace(target)
    return target


def probe(work,source,sas):
    cols=pd.read_csv(source/'detections.csv',nrows=0).columns
    needed=['survey','detection_id','exposure_id','detector_id','field_id','ra_deg','dec_deg','observation_time']
    data=pd.read_csv(source/'detections.csv',usecols=[c for c in needed if c in cols],low_memory=False)
    data=data.loc[data.survey.eq('SDSS')].copy()
    if 'field_id' not in data:data['field_id']=pd.to_numeric(data.detection_id.str.split(':').str[-2],errors='coerce')
    data['run']=data.exposure_id.str.removeprefix('SDSS:') if hasattr(data.exposure_id.str,'removeprefix') else data.exposure_id.str.replace('^SDSS:','',regex=True)
    chosen=data[['run','detector_id','field_id']].dropna().drop_duplicates().head(2)
    results=[];private=[]
    for run,camcol,field in chosen.itertuples(index=False,name=None):
        run,camcol,field=map(int,(run,camcol,field))
        rel=Path('photoObj')/'301'/str(run)/str(camcol)/f'photoObj-{run:06d}-{camcol}-{field:04d}.fits'
        path=file_at(work,sas,rel)
        with fits.open(path,memmap=True) as hdus:
            table=hdus[1].data;lookup=arrays(table)
            required=['RA','DEC','RAERR','DECERR','TAI','OFFSETRA','OFFSETDEC','ID']
            if not set(required)<=lookup.keys():raise ValueError('Incomplete published position/time schema')
            value={k:np.array(table[lookup[k]]) for k in required}
            out=dict(photoobj_sha256=digest(path),photoobj_schema=schema(hdus[1]),band_order=list(BANDS),band_diagnostics={})
            valid=np.isfinite(value['TAI'])&(value['TAI']>4e9)&(value['TAI']<6e9)
            for i,band in enumerate(BANDS):
                use=valid[:,i]&valid[:,2]&np.isfinite(value['OFFSETRA'][:,i])&np.isfinite(value['OFFSETDEC'][:,i])
                dt=value['TAI'][use,i]-value['TAI'][use,2]
                offsets=1000*np.stack([value['OFFSETRA'][use,i],value['OFFSETDEC'][use,i]],axis=1)
                stats=dict(valid_time_offset_count=int(use.sum()),delta_tai_from_r_p16_p50_p84_seconds=np.quantile(dt,[.16,.5,.84]) if len(dt) else None,
                    angular_offset_from_reference_p68_mas=np.quantile(np.linalg.norm(offsets,axis=1),.68) if len(offsets) else None)
                for key in ('ROWCERR','COLCERR','PSFMAGERR'):
                    if key in lookup:
                        v=np.array(table[lookup[key]][:,i],float);v=v[np.isfinite(v)&(v>0)]
                        stats[key.lower()+'_p16_p50_p84']=np.quantile(v,[.16,.5,.84]) if len(v) else None
                out['band_diagnostics'][band]=stats
            # Verify the existing r-only coordinates and times on exact physical IDs.
            subset=data.loc[data.run.astype(int).eq(run)&data.detector_id.eq(camcol)&data.field_id.eq(field)]
            ids=np.array([f'SDSS:{run}:{camcol}:{field}:{int(v)}' for v in value['ID']])
            matched=pd.Index(ids).get_indexer(subset.detection_id)
            if np.any(matched<0):raise ValueError('Frozen source not found in its physical single-scan field')
            use=np.isfinite(subset.observation_time.to_numpy())
            diff=value['TAI'][matched[use],2]
            out['frozen_r_replay']=dict(compared_count=int(use.sum()),maximum_time_difference_days=float(np.max(abs(sdss_tai_to_tt(diff)-subset.observation_time.to_numpy()[use]))) if use.any() else None,
                maximum_ra_difference_deg=float(np.max(abs(value['RA'][matched]-subset.ra_deg.to_numpy()))) if len(matched) else None,
                maximum_dec_difference_deg=float(np.max(abs(value['DEC'][matched]-subset.dec_deg.to_numpy()))) if len(matched) else None)
        rel=Path('photoObj')/'301'/str(run)/f'photoField-{run:06d}-{camcol}.fits'
        try:
            field_path=file_at(work,sas,rel)
            with fits.open(field_path,memmap=True) as hdus:
                table=hdus[1].data;lookup=arrays(table);rows=np.flatnonzero(table[lookup['FIELD']]==field)
                if len(rows)!=1:raise ValueError('Expected one physical calibration field')
                out['photofield_sha256']=digest(field_path);out['photofield_schema']=schema(hdus[1])
                useful=[k for k in lookup if k in ('NODE','INCL','MUERR','NUERR','REF_TAI','TAI','A','B','C','D','E','F','G0','G1','G2','G3','H0','H1','H2','H3','PX','PY','QX','QY','COLOR0')]
                private.append({key:np.array(table[lookup[key]][rows[0]]) for key in useful})
                out['actual_calibration_arrays']={key:dict(shape=list(np.shape(table[lookup[key]][rows[0]])),finite_count=int(np.isfinite(table[lookup[key]][rows[0]]).sum())) for key in useful}
        except (OSError,requests.RequestException,ValueError) as exc:
            out['photofield_status']='unavailable:'+type(exc).__name__
        results.append(out)
    save_json(work/'private_calibration_arrays.json',private)
    output=dict(physical_fields_tested=len(results),fields=results,slurm_job=os.environ['SLURM_JOB_ID'],
        status='schema_time_and_frozen_r_replay_probe_only',
        independence='Five passbands within one drift scan share astrometric calibration; do not count as independent years.',
        adoption='Other SDSS bands not yet approved: per-band centroid plus field WCS error propagation and held-source validation still required. Offsets describe local angular distances; no reuse of scalar RAERR as five independent errors.')
    check_aggregate(output);save_json(work/'sdss_band_probe_summary.json',output)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('work','source','sas','config'):p.add_argument(name,type=Path)
    a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use a Slurm compute node')
    a.work.mkdir(parents=True,exist_ok=True)
    bind(a.work,'source_receipt.json',verify_directory(a.source,('detections.csv',)))
    with run_stage(a.work,a.config,'actual_sdss_fields','sdss-band-probe-v14',inputs=('source_receipt.json',),outputs=('sdss_band_probe_summary.json','private_calibration_arrays.json')) as go:
        if go:probe(a.work,a.source,a.sas)


if __name__=='__main__':main()
