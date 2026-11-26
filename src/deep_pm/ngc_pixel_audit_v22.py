"""Frozen forty: official SDSS masks, local backgrounds and real run support.

Decodes mask pixels using the unmodified official readAtlasImages v5_4_11
read_mask binary installed separately on PScratch; no source extraction/PSF fit.
"""
import argparse
from collections import Counter
from functools import lru_cache
import gzip
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from .astrometry import project
from .blind_catalogue_quality import read
from .coordinate_metric import AngularTree
from .io import digest,save_json
from .public_summary import check_aggregate

PLANES=('INTERP','SATUR','NOTCHECKED','OBJECT','BRIGHTOBJECT','BINOBJECT','CATOBJECT','SUBTRACTED','GHOST','CR')
BAD=('INTERP','SATUR','NOTCHECKED','BRIGHTOBJECT','GHOST','CR')


def run(root,output):
    output.mkdir(parents=True,exist_ok=True);base=root/'v2.2';initial=base/'ngc-early';frozen=read(initial/'private_full_catalogue_opportunities.json');receipts=read(initial/'private_file_receipts.json')
    masks=read(base/'ngc-mask-rerun301/private_mask_paths.json');mask_paths={(r['run'],r['camcol'],r['field']):Path(r['paths'][0]) for r in masks if len(r['paths'])==1}
    readers=list((base/'ngc-mask-reader').glob('**/read_mask'))
    if len(readers)!=1:raise ValueError('Need the separately built official mask reader')
    reader=readers[0];proof=read(base/'ngc-mask-reader/source_receipt.json');decoder_sha=digest(reader)
    input_files={};decoding=[]
    @lru_cache(maxsize=4)
    def load(key):
        run,camcol,field=key;source=mask_paths[key];tag=f'{run:06d}-{camcol}-{field:04d}';folder=output/'decoded'/tag;folder.mkdir(parents=True,exist_ok=True)
        raw=folder/'mask.fit'
        if not raw.exists():
            if source.suffix=='.gz':
                with gzip.open(source,'rb') as i,raw.open('wb') as o:shutil.copyfileobj(i,o)
            else:shutil.copyfile(source,raw)
        with fits.open(raw,memmap=False) as h:
            header=h[0].header;shape=(header['MASKROWS'],header['MASKCOLS'])
            for k,v in [('RUN',run),('CAMCOL',camcol),('FIELD',field)]:
                if k in header and header[k]!=v:raise ValueError('Wrong mask physical field')
            counts=[int(np.sum(t.data['npix'])) if t.data is not None else 0 for t in h[1:11]]
        bits=np.zeros(shape,np.uint16);checks=[]
        for index,name in enumerate(PLANES):
            file=folder/(name+'.fits')
            if not file.exists():subprocess.run([str(reader),str(raw),str(index+1),str(file)],check=True,capture_output=True)
            with fits.open(file,memmap=False) as h:plane=np.asarray(h[0].data)
            if plane.shape!=shape or not np.isin(plane,[0,1]).all():raise ValueError('Invalid official mask plane dimensions/values')
            pixels=int(plane.sum())
            if pixels>counts[index]:raise ValueError('Mask decoder exceeds declared mask pixel count')
            bits[plane!=0]|=np.uint16(1<<index);checks.append(dict(plane=name,declared_pixel_sum=counts[index],decoded_union_pixels=pixels))
        frame_name=f'frame-r-{run:06d}-{camcol}-{field:04d}.fits.bz2';cat_name=f'photoObj-{run:06d}-{camcol}-{field:04d}.fits'
        frame_path=next(Path(p) for p in receipts['frames'] if Path(p).name==frame_name);cat_path=next(Path(p) for p in receipts['catalogues'] if Path(p).name==cat_name)
        for p,expected in [(frame_path,receipts['frames'][str(frame_path)]),(cat_path,receipts['catalogues'][str(cat_path)])]:
            if digest(p)!=expected:raise ValueError('Frozen frame/catalogue changed')
        with fits.open(frame_path,memmap=False) as h:pixel=np.asarray(h[0].data,float);wcs=WCS(h[0].header)
        if pixel.shape!=shape:raise ValueError('Mask and published frame pixel grids differ; cannot certify alignment')
        with fits.open(cat_path,memmap=True) as h:
            t=h[1].data;valid=np.isfinite(t['RA'])&np.isfinite(t['DEC']);sky=np.column_stack([t['RA'][valid],t['DEC'][valid]])
        input_files[str(source)]=digest(source);decoding.append(dict(physical_field_tag=tag,mask_pixel_checks=checks))
        return pixel,wcs,bits,sky
    entries=[];bad_bits=sum(1<<PLANES.index(k) for k in BAD);background_bits=bad_bits|(1<<PLANES.index('OBJECT'))
    for target in frozen:
        visits=[]
        for visit in target['visits']:
            key=(visit['run'],visit['camcol'],visit['field']);item=dict(run=key[0],camcol=key[1],field=key[2]);pixel,wcs,bits,sky=load(key)
            x,y=wcs.all_world2pix([[target['ra'],target['dec']]],0)[0];xx,yy=int(round(x)),int(round(y));edge=min(x,y,pixel.shape[1]-1-x,pixel.shape[0]-1-y)
            if edge<15:visits.append(dict(item,status='insufficient_edge_background',edge_pixels=float(edge)));continue
            cut=pixel[yy-15:yy+16,xx-15:xx+16];flag=bits[yy-15:yy+16,xx-15:xx+16];gy,gx=np.indices(cut.shape);radius=np.hypot(gx-(x-xx+15),gy-(y-yy+15));ap=radius<=2.5;expanded=radius<=3.5
            bg=(radius>=8)&(radius<=14)&np.isfinite(cut)&((flag&background_bits)==0)
            plane={name:int(np.count_nonzero(flag[expanded]&(1<<i))) for i,name in enumerate(PLANES)}
            contaminated=bool(np.any((flag[expanded]&bad_bits)!=0));finite=bool(np.isfinite(cut[expanded]).all())
            item.update(status='masked_or_nonfinite_aperture' if contaminated or not finite else 'conditional_clean_pixel_opportunity',edge_pixels=float(edge),
                conservative_3p5pixel_mask_counts=plane,background_unmasked_pixels=int(bg.sum()),aperture_pixels=int(ap.sum()))
            if bg.sum()>=100 and finite and not contaminated:
                background=float(np.median(cut[bg]));noise=float(1.4826*np.median(abs(cut[bg]-background)));flux=float(np.sum(cut[ap]-background));variance=float(noise**2*ap.sum())
                if variance>0:item.update(flux=flux,variance_proxy=variance,SNR_proxy=flux/np.sqrt(variance))
            center=(target['ra'],target['dec']);dist=np.linalg.norm(project(*sky.T,center),axis=1);near=np.sort(dist[dist<=10000]);exclude_first=bool(len(near) and near[0]<=1000)
            others=near[1:] if exclude_first else near
            item.update(catalogue_sources_within3arcsec=int(np.sum(near<=3000)),nearest_catalogue_distance_mas=float(near[0]) if len(near) else None,
                neighbor_within3arcsec_excluding_possible1arcsec_counterpart=bool(len(others) and others[0]<=3000))
            visits.append(item)
        # Pick geometry before flux/SNR; one published field per physical run,
        # so overlap/rerun duplicates never count as independent early epochs.
        perrun={}
        for v in visits:
            run=v['run']
            if run not in perrun or v['edge_pixels']>perrun[run]['edge_pixels']:perrun[run]=v
        good=[v for v in perrun.values() if 'SNR_proxy' in v];joint=sum(v['flux'] for v in good)/np.sqrt(sum(v['variance_proxy'] for v in good)) if good else None
        entries.append(dict(category=target['category'],frozen_index=target['frozen_index'],r0=target['r0'],visits=visits,
            independent_physical_runs=len(perrun),conditionally_usable_runs=len(good),joint_aperture_SNR_proxy=joint))
    save_json(output/'private_forty_masked_pixels.json',entries);save_json(output/'private_mask_decoding_receipt.json',dict(input_files=input_files,decoder_sha256=decoder_sha,source=proof,decoding=decoding))
    categories={}
    for category in sorted({r['category'] for r in entries}):
        part=[r for r in entries if r['category']==category];visits=[v for r in part for v in r['visits']];snr=[v['SNR_proxy'] for v in visits if 'SNR_proxy' in v];joint=[r['joint_aperture_SNR_proxy'] for r in part if r['joint_aperture_SNR_proxy'] is not None]
        categories[category]=dict(targets=len(part),pixel_status=dict(Counter(v['status'] for v in visits)),
            any_conservative_bad_pixel=sum(any(v['status']=='masked_or_nonfinite_aperture' for v in r['visits']) for r in part),
            any_published_neighbor3arcsec=sum(any(v.get('neighbor_within3arcsec_excluding_possible1arcsec_counterpart') for v in r['visits']) for r in part),
            conditional_SNR_target_count=len(joint),per_exposure_SNR_proxy_p16_p50_p84=np.quantile(snr,[.16,.5,.84]) if snr else None,
            one_per_run_joint_SNR_proxy_p16_p50_p84=np.quantile(joint,[.16,.5,.84]) if joint else None,
            multiple_conditionally_usable_runs=sum(r['conditionally_usable_runs']>=2 for r in part))
    result=dict(categories=categories,unique_decoded_physical_fields=len(mask_paths),official_reader_source=proof,decoder_sha256=decoder_sha,
        frozen_forty_sha256=digest(initial/'private_frozen_forty.json'),slurm_job=os.environ['SLURM_JOB_ID'],
        scope='Same40 photometric targets, no PM selection. Real rerun301/run fpM exact mask planes decoded by official reader; mask/frame grid and input hashes checked. 3.5pixel conservative bad region around2.5pixel aperture; background removes catalogued object mask. Geometry chooses one field per physical run. SNR proxies ignore remaining pixel covariance, source Poisson/variability and centroid bias; no new position/proper motion fitted. 1arcsec photometric-position catalogue absence is not a proof of source-extraction failure when physical identity/motion is unknown. Neighbor flags are conditional catalogue evidence, not stellar truth.')
    check_aggregate(result);save_json(output/'v22_ngc_masked_pixel_opportunity_summary.json',result)
    print('NGC_FORTY_MASKED_PIXEL_AUDIT_COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use Slurm')
    run(a.root,a.output)
