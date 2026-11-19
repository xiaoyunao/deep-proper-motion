"""Verify published per-band angular offsets against actual single-scan WCS."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

from .astrometry import project
from .continuous_reference import PRECISION, bind, verify_directory
from .provenance import run_stage
from .io import save_json,digest
from .public_summary import check_aggregate
from .sdss_band_probe import arrays,file_at,BANDS


def great_circle_icrs(mu,nu,node,incl):
    a,d,i=np.deg2rad([mu-node,nu,np.broadcast_to(incl,np.shape(mu))])
    x=np.cos(a)*np.cos(d)
    y=np.sin(a)*np.cos(d)*np.cos(i)-np.sin(d)*np.sin(i)
    z=np.sin(a)*np.cos(d)*np.sin(i)+np.sin(d)*np.cos(i)
    return (node+np.rad2deg(np.arctan2(y,x)))%360,np.rad2deg(np.arctan2(z,np.hypot(x,y)))


def wcs_position(row,col,color,field,band,node,incl):
    def c(name):return float(field[name][band])
    x=row+sum(c('DROW'+str(k))*col**k for k in range(4))
    y=col+sum(c('DCOL'+str(k))*col**k for k in range(4))
    blue=color<c('RICUT')
    x+=np.where(blue,c('CSROW')*color,c('CCROW'))
    y+=np.where(blue,c('CSCOL')*color,c('CCCOL'))
    return great_circle_icrs(c('A')+c('B')*x+c('C')*y,c('D')+c('E')*x+c('FF')*y,node,incl)


def run(work,probe_work,sas):
    # Probe cache is used only as a list of already read physical fields, never for source identities.
    source=(probe_work/'source_receipt.json').resolve().parent
    data=pd.read_csv(source/'detections.csv',usecols=['survey','detection_id','exposure_id','detector_id','field_id'],low_memory=False)
    chosen=data.loc[data.survey.eq('SDSS'),['exposure_id','detector_id','field_id']].drop_duplicates().head(2)
    output=[]
    for exp,camcol,number in chosen.itertuples(index=False,name=None):
        run_id=int(exp.split(':')[1]);camcol=int(camcol);number=int(number)
        a=file_at(probe_work,sas,Path('photoObj')/'301'/str(run_id)/str(camcol)/f'photoObj-{run_id:06d}-{camcol}-{number:04d}.fits')
        b=file_at(probe_work,sas,Path('photoObj')/'301'/str(run_id)/f'photoField-{run_id:06d}-{camcol}.fits')
        with fits.open(a,memmap=True) as obs,fits.open(b,memmap=True) as cal:
            table=obs[1].data;field_table=cal[1].data
            lookup=arrays(field_table);index=np.flatnonzero(field_table[lookup['FIELD']]==number)
            if len(index)!=1:raise ValueError('Physical field calibration is not unique')
            field={k:np.asarray(field_table[v][index[0]]) for k,v in lookup.items()}
            header=dict(cal[0].header);header.update(dict(cal[1].header))
            node=header.get('NODE');incl=header.get('INCL')
            if node is None or incl is None:raise ValueError('Scan NODE/INCL missing from actual photoField headers')
            names=arrays(table);ra=np.asarray(table[names['RA']],float);dec=np.asarray(table[names['DEC']],float)
            centers=(float(np.median(ra)),float(np.median(dec)))
            offsets=np.stack([table[names['OFFSETRA']],table[names['OFFSETDEC']]],axis=-1)
            stats={}
            for j,band in enumerate(BANDS):
                stats[band]={}
                for colour_key in ('PSFMAG','MODELMAG'):
                    if colour_key not in names:continue
                    mag=np.asarray(table[names[colour_key]],float)
                    ci,cj=(0,1) if j==0 else (1,2) if j==1 else (2,3)
                    color=mag[:,ci]-mag[:,cj]
                    row=np.asarray(table[names['ROWC']][:,j],float);col=np.asarray(table[names['COLC']][:,j],float)
                    err=np.stack([table[names['ROWCERR']][:,j],table[names['COLCERR']][:,j]],axis=1)
                    valid=np.isfinite(row+col+color)&np.isfinite(err).all(axis=1)&(err>0).all(axis=1)&(mag[:,ci]>0)&(mag[:,ci]<35)&(mag[:,cj]>0)&(mag[:,cj]<35)
                    angular=np.column_stack([(ra+offsets[:,j,0]/3600/np.cos(np.deg2rad(dec)))%360,dec+offsets[:,j,1]/3600])
                    valid &= np.isfinite(angular).all(axis=1)&(abs(angular[:,1])<90)
                    if not valid.any():continue
                    target=project(*angular[valid].T,centers)
                    variants={}
                    for label,sign in [('frame_pixels',0),('add_frame_offset',1),('subtract_frame_offset',-1)]:
                        x=row[valid]+sign*float(field.get('ROWOFFSET',np.zeros(5))[j]);y=col[valid]+sign*float(field.get('COLOFFSET',np.zeros(5))[j])
                        sky=wcs_position(x,y,color[valid],field,j,node,incl)
                        residual=project(*sky,centers)-target
                        variants[label]=dict(n=int(valid.sum()),median_mas=np.median(residual,axis=0),radial_p68_mas=float(np.quantile(np.linalg.norm(residual,axis=1),.68)))
                    stats[band][colour_key]=variants
            output.append(dict(photoobj_sha256=digest(a),photofield_sha256=digest(b),band_wcs_offset_reconstruction=stats))
    result=dict(physical_fields=len(output),fields=output,slurm_job=os.environ['SLURM_JOB_ID'],
        scope='Position-only reconstruction from published row/column, optical-distortion/DCR and affine coefficients; compares exact published per-band angular offsets. No proper-motion truth, no extra astrometric correction. Scalar r errors are not cloned.',
        adoption='Not approved for joint fitting yet: centroid/WCS covariance, shared-scan errors, per-band detection flags and independent held-object validation still required.')
    check_aggregate(result);save_json(work/'sdss_position_check_summary.json',result)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('work','probe_work','sas'):p.add_argument(n,type=Path)
    a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use a Slurm compute node')
    a.work.mkdir(parents=True,exist_ok=True)
    bind(a.work,'probe_receipt.json',verify_directory(a.probe_work,('sdss_band_probe_summary.json','private_calibration_arrays.json')))
    with run_stage(a.work,PRECISION,'actual_sdss_fields','sdss-position-check-v14',inputs=('probe_receipt.json',),outputs=('sdss_position_check_summary.json',)) as go:
        if go:run(**vars(a))


if __name__=='__main__':main()
