"""Verify published per-band angular offsets against actual single-scan WCS."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

from .astrometry import project
from .footprints import sdss_great_circle,wrap
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


def position_covariance(row_error,col_error,col,field,band,sky,node,incl):
    """Native centroid pixels and per-field great-circle calibration errors.

    Sky RAERR is already an angular east error: its comparison below never
    multiplies it by cos(dec). The cos(dec) here belongs only to the coordinate
    Jacobian of a perturbed RA coordinate, not to published errors.
    """
    ra,dec=sky;mu,nu=sdss_great_circle(ra,dec,node,incl);step=1e-5
    rotation=np.empty((len(ra),2,2))
    for axis in range(2):
        a,d=great_circle_icrs(mu+(step if axis==0 else 0),nu+(step if axis==1 else 0),node,incl)
        rotation[:,0,axis]=wrap(a-ra)*np.cos(np.deg2rad(dec))/step
        rotation[:,1,axis]=(d-dec)/step
    def c(key):return float(field[key][band])
    dr=c('DROW1')+2*c('DROW2')*col+3*c('DROW3')*col**2
    dc=1+c('DCOL1')+2*c('DCOL2')*col+3*c('DCOL3')*col**2
    pixel=np.empty_like(rotation);pixel[:,0,0]=c('B');pixel[:,0,1]=c('B')*dr+c('C')*dc
    pixel[:,1,0]=c('E');pixel[:,1,1]=c('E')*dr+c('FF')*dc
    pixel=np.einsum('nab,nbc->nac',rotation,pixel)*3600000
    centroid=np.einsum('nab,nb,ncb->nac',pixel,np.column_stack([row_error**2,col_error**2]),pixel)
    calibration=np.einsum('nab,b,ncb->nac',rotation,np.array([c('MUERR'),c('NUERR')])**2*1e6,rotation)
    return centroid,calibration


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
                stats[band]={'coefficient_roundoff_half_ulp_mas':{k:float(abs(np.spacing(np.float32(field[k][j]))))*3600000/2 for k in ('A','D')}}
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
                    if colour_key=='PSFMAG':
                        nominal=wcs_position(row[valid],col[valid],color[valid],field,j,node,incl)
                        centroid,calibration=position_covariance(err[valid,0],err[valid,1],col[valid],field,j,nominal,node,incl)
                        total=centroid+calibration;std=np.sqrt(np.diagonal(total,axis1=1,axis2=2))
                        rho=total[:,0,1]/(std[:,0]*std[:,1])
                        stats[band]['native_error_propagation']=dict(valid_count=len(std),
                            centroid_sigma_p16_p50_p84_mas=np.quantile(np.sqrt(np.diagonal(centroid,axis1=1,axis2=2)),[.16,.5,.84],axis=0),
                            calibration_sigma_p16_p50_p84_mas=np.quantile(np.sqrt(np.diagonal(calibration,axis1=1,axis2=2)),[.16,.5,.84],axis=0),
                            total_sigma_p16_p50_p84_mas=np.quantile(std,[.16,.5,.84],axis=0),cross_axis_correlation_p16_p50_p84=np.quantile(rho,[.16,.5,.84]))
                        if j==2:
                            published=np.column_stack([table[names['RAERR']][valid],table[names['DECERR']][valid]])*1000
                            reference_r=(offsets[valid,2]==0).all(axis=1)&np.isfinite(published).all(axis=1)&(published>0).all(axis=1)
                            if reference_r.any():
                                stats[band]['native_error_propagation']['rowc_total_to_published_r_error_ratio_p16_p50_p84']=np.quantile(std[reference_r]/published[reference_r],[.16,.5,.84],axis=0)
                                if 'OBJC_ROWCERR' in names and 'OBJC_COLCERR' in names:
                                    objc=np.column_stack([table[names['OBJC_ROWCERR']][valid],table[names['OBJC_COLCERR']][valid]])
                                    good=reference_r&np.isfinite(objc).all(axis=1)&(objc>0).all(axis=1)
                                    if good.any():
                                        ce,ca=position_covariance(objc[good,0],objc[good,1],col[valid][good],field,j,(nominal[0][good],nominal[1][good]),node,incl)
                                        stats[band]['native_error_propagation']['objc_total_to_published_r_error_ratio_p16_p50_p84']=np.quantile(np.sqrt(np.diagonal(ce+ca,axis1=1,axis2=2))/published[good],[.16,.5,.84],axis=0)

                    variants={}
                    for label,sign in [('frame_pixels',0),('add_frame_offset',1),('subtract_frame_offset',-1)]:
                        x=row[valid]+sign*float(field.get('ROWOFFSET',np.zeros(5))[j]);y=col[valid]+sign*float(field.get('COLOFFSET',np.zeros(5))[j])
                        sky=wcs_position(x,y,color[valid],field,j,node,incl)
                        residual=project(*sky,centers)-target
                        median=np.median(residual,axis=0)
                        variants[label]=dict(n=int(valid.sum()),median_mas=median,radial_p68_mas=float(np.quantile(np.linalg.norm(residual,axis=1),.68)),
                            recentered_radial_p68_mas=float(np.quantile(np.linalg.norm(residual-median,axis=1),.68)),
                            nmad_mas=1.4826*np.median(abs(residual-median),axis=0))
                    stats[band][colour_key]=variants
            output.append(dict(photoobj_sha256=digest(a),photofield_sha256=digest(b),band_wcs_offset_reconstruction=stats))
    result=dict(physical_fields=len(output),fields=output,slurm_job=os.environ['SLURM_JOB_ID'],
        coefficient_precision_note='A/D are actual float32 FITS 5E coefficients. Their half-ULP gives a possible constant great-circle-coordinate roundoff, not a target centroid error. Recentered residual is a diagnostic, not a new astrometric correction.',
        scope='Position-only reconstruction from published row/column, optical-distortion/DCR and affine coefficients; compares exact published per-band angular offsets. No proper-motion truth, no extra astrometric correction. Scalar r errors are not cloned.',
        native_error_scope='Centroid covariance assumes independent native row/column errors and fixed color; field MUERR/NUERR treated as great-circle coordinate errors in arcsec. Published r error ratios test the convention. Cross-band shared scan/calibration covariance and color-error propagation remain unverified.',
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
