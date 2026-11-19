"""Inventory every published filter without selecting a band-name whitelist."""
import argparse
import json
import os
from pathlib import Path

from astropy.time import Time
import numpy as np
import pandas as pd

from .__main__ import tomllib
from .continuous_instruments import chip_receipts,instrument_by_exposure
from .continuous_reference import bind,read_offsets,verify_directory
from .exposure_catalogues import inside
from .io import digest,save_json
from .precision_v03 import calibration_groups,calibration_support
from .provenance import run_stage
from .public_summary import check_aggregate

EXPECTED=('u','g','r','i','z','Y','VR')  # Report missing bands, never a selection.


def quantiles(values):
    x=np.asarray(values,float);x=x[np.isfinite(x)]
    return np.quantile(x,[.16,.5,.84]) if len(x) else None


def summarize(data,offsets,spec):
    t=data.loc[inside(data.ra_deg,data.dec_deg,spec)].copy().reset_index(drop=True)
    keys=calibration_groups(t);support=calibration_support(keys,offsets)
    native=t[['sigma_east_mas','sigma_north_mas']].to_numpy(float)
    sky=t[['ra_deg','dec_deg','observation_time']].to_numpy(float)
    legal=np.isfinite(sky).all(axis=1)&np.isfinite(native).all(axis=1)&(native>0).all(axis=1)&(native<=200).all(axis=1)
    legal &= t.dec_deg.between(-90,90).to_numpy()&t.ra_deg.between(0,360,inclusive='left').to_numpy()
    magerr=pd.to_numeric(t.magnitude_error,errors='coerce').to_numpy(float)
    snr=np.divide(2.5/np.log(10),magerr,out=np.full(len(t),np.nan),where=magerr>0)
    nsc=t.survey.eq('NSC').to_numpy();rawflags=pd.to_numeric(t.quality_flags,errors='coerce').to_numpy()
    validflags=np.isfinite(rawflags)&(rawflags>=0)&(rawflags<=255)
    flags=np.where(validflags,rawflags,0).astype(np.int64)
    clean=validflags&((flags&252)==0)
    candidate=legal&(snr>=5)&((nsc&clean)|~nsc)
    year=np.full(len(t),-1,int);night=np.full(len(t),-1,int)
    finite=np.isfinite(sky[:,2])
    if finite.any():
        unique,inverse=np.unique(sky[finite,2],return_inverse=True)
        utc=Time(unique,format='mjd',scale='tt').utc
        year[finite]=np.array([int(s[:4]) for s in utc.isot])[inverse]
        night[finite]=np.floor(utc.mjd[inverse]-.75).astype(int)
    result={}
    group_columns=['survey','instrument','band']
    for label,part in t.groupby(group_columns,dropna=False,sort=True):
        ix=part.index.to_numpy();valid=ix[legal[ix]];chosen=ix[candidate[ix]]
        out=dict(measurement_count=len(ix),distinct_exposures=part.exposure_id.nunique(),
            distinct_exposure_ccds=part[['exposure_id','detector_id']].drop_duplicates().shape[0],
            independent_utc18_nights=len(set(night[ix]) - {-1}),utc_year_counts={str(y):int((year[ix]==y).sum()) for y in sorted(set(year[ix]) - {-1})},
            native_position_error_p16_p50_p84_mas=[quantiles(native[ix,a]) for a in range(2)],
            photometric_snr_p16_p50_p84=quantiles(snr[ix]),
            valid_native_time_position_count=len(valid),reference_supported_count=int(support[ix].sum()),
            snr_flags_position_candidate_count=len(chosen),candidate_reference_supported_count=int(support[chosen].sum()),
            candidate_nights=len(set(night[chosen]) - {-1}),
            flags_unknown_count=int((~validflags[ix]).sum()) if label[0]=='NSC' else None,
            sextractor_deblended_count=int(((flags[ix]&2)!=0).sum()) if label[0]=='NSC' else None,
            sextractor_fatal_count=int((validflags[ix]&((flags[ix]&252)!=0)).sum()) if label[0]=='NSC' else None)
        if len(valid):out['observed_span_years']=float(np.ptp(sky[valid,2])/365.25)
        for column in ('x','y','fwhm','airmass','color','field_id'):
            values=pd.to_numeric(part[column],errors='coerce') if column in part else pd.Series(np.nan,index=part.index)
            out['finite_'+column+'_count']=int(np.isfinite(values).sum())
        result[':'.join(map(str,label))]=out
    actual=set(t.loc[nsc,'band'].dropna().astype(str))
    return dict(core_detection_count=len(t),duplicate_detection_count=int(t.detection_id.duplicated().sum()),
        instrument_band=result,expected_nsc_bands_absent=sorted(set(EXPECTED)-actual),
        actual_nsc_bands=sorted(actual),unexpected_nsc_band_labels=sorted(actual-set(EXPECTED)),
        scope='Counts on the declared core; all retrieved filter labels retained. Candidate counts use finite native errors <=200mas, valid TT time, SNR>=5 and NSC fatal flags clear; this is NOT validated high-quality astrometry. WCS, blending, cross-band bias and shared-error validation remain separate. SDSS legacy scalar flags are not promoted to per-band quality.')


def run(work,source,shared,spec):
    header=pd.read_csv(source/'detections.csv',nrows=0).columns
    wanted=['survey','detection_id','exposure_id','detector_id','ra_deg','dec_deg','observation_time','sigma_east_mas','sigma_north_mas','band','magnitude_error','quality_flags','instrument','x','y','fwhm','airmass','color','field_id']
    data=pd.read_csv(source/'detections.csv',usecols=[k for k in wanted if k in header],low_memory=False)
    chips,complete=chip_receipts((source/'run_manifest.json').resolve())
    instruments=instrument_by_exposure(chips)
    if 'instrument' not in data:data['instrument']=np.nan
    data['instrument']=data.instrument.fillna(data.exposure_id.map(instruments)).fillna('unknown')
    data.loc[data.survey.eq('SDSS'),'instrument']='SDSS_imager'
    result=summarize(data,read_offsets(shared/'private_c0.npz'),spec)
    result.update(chip_metadata_receipts_complete=complete,slurm_job=os.environ['SLURM_JOB_ID'],
        shared_solution_sha256=digest(shared/'private_c0.npz'),source_receipt_sha256=digest(source/'run_manifest.json'))
    check_aggregate(result);save_json(work/'allband_inventory_summary.json',result)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('work','source','shared','config'):p.add_argument(name,type=Path)
    p.add_argument('region');a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use a Slurm compute node')
    a.work.mkdir(parents=True,exist_ok=True)
    bind(a.work,'source_receipt.json',verify_directory(a.source,('detections.csv',)))
    bind(a.work,'reference_receipt.json',verify_directory(a.shared,('private_c0.npz',)))
    with run_stage(a.work,a.config,a.region,'allband-inventory-v14',inputs=('source_receipt.json','reference_receipt.json'),outputs=('allband_inventory_summary.json',)) as go:
        if go:run(a.work,a.source,a.shared,tomllib.loads(a.config.read_text())['fields'][a.region])


if __name__=='__main__':main()
