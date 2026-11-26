"""Forty frozen optical targets: exact early SDSS fields and full catalogues.

No PM selection, no catalogue production, no source extraction. Optional frames
provide conditional pixel/SNR diagnostics; image-mask certification stays explicit.
"""
import argparse
from collections import Counter
from functools import lru_cache
from hashlib import sha256
import os
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS

from .astrometry import project
from .blind_catalogue_quality import read
from .coordinate_metric import AngularTree
from .footprints import sdss_great_circle,sdss_window_mask
from .io import digest,save_json
from .public_summary import check_aggregate
from .sdss_band_probe import file_at


def hash_key(value):return sha256(('v22-ngc-early:'+str(value)).encode()).hexdigest()


def run(root,sas,output,images=False):
    output.mkdir(parents=True,exist_ok=True)
    phot_path=root/'v2.0/ngc_photometry/private_photometric_parent.csv.gz'
    traces_path=root/'v2.1/ngc415/private_ngc415_trace.json.gz'
    phot=pd.read_csv(phot_path);traces=read(traces_path)
    categories=[('no_cached_SDSS','no_published_SDSS_detection',20),('SDSS_quality_distance','SDSS_quality_or_distance',10),('positive_early_control','recovered',10)]
    selected=[]
    for category,status,number in categories:
        pool=sorted([r for r in traces if r['statuses']['geometry']['primary_exit']==status],key=lambda r:hash_key(r['frozen_index']))
        if len(pool)<number:raise ValueError('Insufficient frozen category support')
        for trace in pool[:number]:
            target=phot.iloc[trace['frozen_index']]
            selected.append(dict(frozen_index=trace['frozen_index'],category=category,ra=float(target.ra),dec=float(target.dec),r0=float(target.r0),core=trace['core']))
    save_json(output/'private_frozen_forty.json',dict(photometry_sha256=digest(phot_path),trace_sha256=digest(traces_path),targets=selected))
    window=sas/'resolve/2013-07-29/window_flist.fits'
    with fits.open(window,memmap=True) as h:
        t=h[1].data
        nearby=sdss_window_mask(t,301)&(abs(t['DEC']-np.mean([r['dec'] for r in selected]))<.6)
        dra=(t['RA']-np.mean([r['ra'] for r in selected])+180)%360-180
        nearby&=abs(dra)<.8
        # Astropy FITS_rec slicing can retain original column-array buffers;
        # detach columns before a second boolean slice on the reduced fields.
        fields={k:np.array(t[k][np.flatnonzero(nearby)],copy=True) for k in ('RA','DEC','RUN','CAMCOL','FIELD','NODE','INCL','MU_START','MU_END','NU_START','NU_END')}
    receipts={};positions=[];frame_receipts={}
    @lru_cache(maxsize=32)
    def catalogue(key):
        run,rerun,camcol,field=key
        relative=Path('photoObj')/str(rerun)/str(run)/str(camcol)/f'photoObj-{run:06d}-{camcol}-{field:04d}.fits'
        file=file_at(output,sas,relative);receipts[str(file)]=digest(file)
        with fits.open(file,memmap=True) as h:
            t=h[1].data
            cols=['RA','DEC','ID','RAERR','DECERR','RESOLVE_STATUS','OBJC_FLAGS','NCHILD','TAI','PSFMAG','PSFMAGERR','PSF_FWHM']
            return {name:np.asarray(t[name]).copy() for name in cols if name in t.names}
    @lru_cache(maxsize=8)
    def image(key):
        run,rerun,camcol,field=key
        relative=Path('photoObj/frames')/str(rerun)/str(run)/str(camcol)/f'frame-r-{run:06d}-{camcol}-{field:04d}.fits.bz2'
        file=file_at(output,sas,relative) if (sas/relative).exists() else None
        if file is None:
            import requests
            file=output/'cache'/relative;file.parent.mkdir(parents=True,exist_ok=True)
            if not file.exists():
                response=requests.get('https://data.sdss.org/sas/dr17/eboss/'+str(relative),timeout=(15,120));response.raise_for_status()
                if not response.content.startswith(b'BZh'):raise ValueError('Expected compressed frame FITS')
                pending=file.with_suffix('.part');pending.write_bytes(response.content);pending.replace(file)
        frame_receipts[str(file)]=digest(file)
        with fits.open(file,memmap=False) as h:
            return np.asarray(h[0].data,float),WCS(h[0].header)
    for target in selected:
        mu,nu=sdss_great_circle(target['ra'],target['dec'],fields['NODE'],fields['INCL'])
        hit=((mu-fields['MU_START'])%360 <= (fields['MU_END']-fields['MU_START'])%360)&(nu>=np.minimum(fields['NU_START'],fields['NU_END']))&(nu<=np.maximum(fields['NU_START'],fields['NU_END']))
        keys=sorted({(int(fields['RUN'][j]),301,int(fields['CAMCOL'][j]),int(fields['FIELD'][j])) for j in np.flatnonzero(hit)})
        visits=[]
        for key in keys:
            entry=dict(run=key[0],camcol=key[2],field=key[3],footprint=True)
            try:
                table=catalogue(key);valid=np.isfinite(table['RA'])&np.isfinite(table['DEC']);ix=np.flatnonzero(valid)
                center=(target['ra'],target['dec']);tree=AngularTree(project(table['RA'][valid],table['DEC'][valid],center),center)
                distance,nearest=tree.query(np.zeros(2));j=ix[int(nearest)]
                entry.update(catalogue_available=True,nearest_distance_mas=float(distance),within_one_arcsec=bool(distance<=1000),
                    psfmag_r=float(table['PSFMAG'][j,2]),psfmagerr_r=float(table['PSFMAGERR'][j,2]),
                    native_sigma_mas=[1000*float(table['RAERR'][j]),1000*float(table['DECERR'][j])],
                    run_primary=bool(int(table['RESOLVE_STATUS'][j])&1),flags=int(table['OBJC_FLAGS'][j]),nchild=int(table['NCHILD'][j]),
                    TAI_r_seconds=float(table['TAI'][j,2]))
            except (OSError,ValueError,KeyError) as error:entry.update(catalogue_available=False,catalogue_failure=type(error).__name__+':'+str(error))
            if images:
                try:
                    pixel,wcs=image(key);x,y=wcs.all_world2pix([[target['ra'],target['dec']]],0)[0];xx,yy=int(round(x)),int(round(y))
                    if xx<15 or yy<15 or xx>=pixel.shape[1]-15 or yy>=pixel.shape[0]-15:raise ValueError('Image edge lacks background annulus')
                    cut=pixel[yy-15:yy+16,xx-15:xx+16];gy,gx=np.indices(cut.shape);rr=np.hypot(gx-(x-xx+15),gy-(y-yy+15))
                    bg=cut[(rr>=8)&(rr<=14)&np.isfinite(cut)];background=float(np.median(bg));noise=float(1.4826*np.median(abs(bg-background)))
                    mask=(rr<=2.5)&np.isfinite(cut);flux=float(np.sum(cut[mask]-background));snr=flux/(noise*np.sqrt(mask.sum())) if noise>0 else None
                    entry.update(frame_available=True,pixel_finite_fraction=float(np.isfinite(cut).mean()),aperture_flux=flux,
                        empirical_background=background,empirical_pixel_noise=noise,aperture_snr_proxy=snr,
                        internal_pixel_mask_verified=False,pixel_diagnostic='2.5pixel aperture,local8to14pixel annulus; correlated noise/blends/source Poisson not fully modelled; not a centroid measurement')
                except (OSError,ValueError,KeyError,RuntimeError) as error:entry.update(frame_available=False,frame_failure=type(error).__name__+':'+str(error))
            visits.append(entry)
        positions.append(dict(**target,visits=visits))
        print('NGC_EARLY_OPPORTUNITY',target['category'],len(keys),flush=True)
    save_json(output/'private_full_catalogue_opportunities.json',positions)
    save_json(output/'private_file_receipts.json',dict(catalogues=receipts,frames=frame_receipts))
    categories={}
    for category,_,_ in categories_spec():
        chosen=[r for r in positions if r['category']==category];visits=[v for r in chosen for v in r['visits']]
        snr=[v['aperture_snr_proxy'] for v in visits if v.get('aperture_snr_proxy') is not None]
        categories[category]=dict(targets=len(chosen),r0_p16_p50_p84=np.quantile([r['r0'] for r in chosen],[.16,.5,.84]),
            with_native_footprint=sum(bool(r['visits']) for r in chosen),with_complete_catalogue=sum(any(v.get('catalogue_available') for v in r['visits']) for r in chosen),
            with_published_within_one_arcsec=sum(any(v.get('within_one_arcsec',False) for v in r['visits']) for r in chosen),
            with_multiple_physical_runs=sum(len({v['run'] for v in r['visits']})>1 for r in chosen),
            with_frame=sum(any(v.get('frame_available') for v in r['visits']) for r in chosen),
            per_exposure_snr_p16_p50_p84=np.quantile(snr,[.16,.5,.84]) if snr else None,
            catalogue_failures=dict(Counter(v.get('catalogue_failure') for v in visits if v.get('catalogue_failure'))),
            frame_failures=dict(Counter(v.get('frame_failure') for v in visits if v.get('frame_failure'))))
    result=dict(frozen_forty_sha256=digest(output/'private_frozen_forty.json'),categories=categories,
        unique_catalogue_files=len(receipts),unique_frame_files=len(frame_receipts),slurm_job=os.environ['SLURM_JOB_ID'],
        scope='Frozen optical identities hash-selected within cache-exit categories; full single-scan photoObj includes all resolve statuses, native great-circle footprints; no Gaia PM or new PM used for sample selection. Pixel aperture diagnostics conditional; bad-pixel mask and forced-position bias require explicit further verification. Positive-control magnitude distribution supplied, not assumed equal.')
    check_aggregate(result);save_json(output/'v22_ngc_early_opportunity_summary.json',result)


def categories_spec():return [('no_cached_SDSS',None,20),('SDSS_quality_distance',None,10),('positive_early_control',None,10)]


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','sas','output'):p.add_argument(name,type=Path)
    p.add_argument('--images',action='store_true');a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use Slurm resources')
    run(**vars(a))
