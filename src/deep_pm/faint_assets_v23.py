"""Acquire only physical-field PSFs and noise/time metadata of frozen frames."""
import argparse,os,re
from pathlib import Path
import numpy as np
from astropy.io import fits
from .blind_catalogue_quality import read
from .faint_preflight_v23 import fetch,run as preflight
from .faint_resources_v23 import run as resources
from .io import digest,save_json
from .public_summary import check_aggregate


def run(base,root):
    resources(root);preflight(base,root/'preflight')
    source=read(base/'v2.2/ngc-early/private_file_receipts.json');out=root/'assets';out.mkdir(exist_ok=True);rows=[]
    for filename,sha in source['frames'].items():
        frame=Path(filename);parts=re.search(r'frame-r-(\d+)-(\d)-(\d+)\.fits',frame.name)
        run,camcol,field=map(int,parts.groups());psfname=f'psField-{run:06d}-{camcol}-{field:04d}.fit'
        url=f'https://data.sdss.org/sas/dr17/eboss/photo/redux/301/{run}/objcs/{camcol}/{psfname}'
        psf=fetch(url,out/'psField');cat=next(Path(p) for p in source['catalogues'] if Path(p).name==f'photoObj-{run:06d}-{camcol}-{field:04d}.fits')
        with fits.open(psf,memmap=False) as h:
            if len(h)<7 or h[3].data is None:raise ValueError('No actual r PSF eigenimages')
            metadata=h[6].data;names={c.upper():c for c in metadata.names};gain=float(metadata[names['GAIN']][0,2]);dark=float(metadata[names['DARK_VARIANCE']][0,2]);status=int(metadata[names['STATUS']][0,2])
            shape=[int(h[3].data['rnrow'][0]),int(h[3].data['rncol'][0])]
        with fits.open(cat,memmap=True) as h:
            t=h[1].data;names={c.upper():c for c in t.names};position=np.asarray(t[names['ROWC']][:,2],float);tai=np.asarray(t[names['TAI']][:,2],float)
            ok=np.isfinite(position)&np.isfinite(tai)&(tai>4e9)&(tai<6e9)
            if ok.sum()<20:raise ValueError('Insufficient published row/time support')
            coef=np.polynomial.polynomial.polyfit(position[ok],tai[ok]-tai[ok].mean(),1);origin=float(tai[ok].mean());residual=tai[ok]-(origin+coef[0]+coef[1]*position[ok])
            if abs(coef[1])<.001 or abs(coef[1])>.2 or np.quantile(abs(residual),.99)>.5:raise ValueError('Published time vs detector row relation not verified')
        with fits.open(frame,memmap=False) as h:
            header={k:str(h[0].header[k]) for k in h[0].header if k not in ('COMMENT','HISTORY','')};calib=np.asarray(h[1].data);sky=h[2].data;astr=h[3].data
            detail=dict(frame=filename,frame_sha256=sha,psField=str(psf),psField_sha256=digest(psf),catalogue=str(cat),catalogue_sha256=source['catalogues'][str(cat)],
                run=run,camcol=camcol,field=field,gain=gain,dark_variance=dark,psf_status=status,psf_kernel_shape=shape,
                tai_r_at_row_zero_seconds=origin+float(coef[0]),tai_r_per_row_seconds=float(coef[1]),time_row_support=int(ok.sum()),time_row_p99_residual_seconds=float(np.quantile(abs(residual),.99)),
                calib_shape=list(calib.shape),sky_columns=list(sky.names),astrometry_columns=list(astr.names),header=header)
            rows.append(detail)
        print('FROZEN_FIELD_ASSET',run,camcol,field,gain,dark,status,coef[1],flush=True)
    save_json(out/'private_field_assets.json',rows)
    summary=dict(actual_fields=len(rows),r_psf_status_counts={str(v):sum(r['psf_status']==v for r in rows) for v in sorted({r['psf_status'] for r in rows})},PSF_metadata=[{k:r[k] for k in ['run','camcol','field','gain','dark_variance','psf_status','psf_kernel_shape','time_row_support','time_row_p99_residual_seconds']} for r in rows],slurm_job=os.environ['SLURM_JOB_ID'],
        scope='Frozen13 physical r fields, official psField spatial eigenPSF, gain/dark and observed per-source row time relation. These are metadata before pixel positions or PM residuals; full asTrans remains available for independent frame-to-published registration. No reuse of public r centroid as a second target likelihood.')
    check_aggregate(summary);save_json(out/'v23_field_assets_summary.json',summary);resources(root)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('base',type=Path);p.add_argument('root',type=Path);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use Slurm')
    run(a.base,a.root)
