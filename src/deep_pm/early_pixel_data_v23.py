"""Frozen-field pixel data, official noise, and independent published-r tie."""
import argparse
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
import os
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from .astrometry import project,unproject,sdss_tai_to_tt,years
from .blind_catalogue_quality import read
from .coordinate_metric import AngularTree,forward_jacobian
from .early_pixel_model_v23 import SpatialPSF,flux_profile,numerical_jacobian
from .io import digest,save_json
from .public_summary import check_aggregate
from .ngc_pixel_audit_v22 import PLANES,BAD


def sky_key(ra,dec):return sha256(('v23-physical-calibrator:'+f'{ra:.5f}:{dec:.5f}').encode()).hexdigest()

def bad_mask(base,a):
    tag=f"{a['run']:06d}-{a['camcol']}-{a['field']:04d}"
    found=[p for p in (base/'v2.2').glob('ngc*/decoded') if (p/tag/'INTERP.fits').exists()]
    if len(found)!=1:raise ValueError('Need unique frozen decoded official mask directory')
    bits=None;objectmask=None
    for label in BAD+('OBJECT',):
        plane=fits.getdata(found[0]/tag/(label+'.fits'))!=0
        if label=='OBJECT':objectmask=plane
        elif bits is None:bits=plane.copy()
        else:bits|=plane
    return bits,objectmask

class Field:
    def __init__(self,base,a):
        self.a=a;self.psf=SpatialPSF(a['psField']);self.bad,self.objects=bad_mask(base,a)
        with fits.open(a['frame'],memmap=False) as h:
            self.pixel=np.asarray(h[0].data,float);self.wcs=WCS(h[0].header);self.calib=np.asarray(h[1].data,float)
            sky=h[2].data;self.sky=np.asarray(sky['ALLSKY'][0],float);self.sx=np.asarray(sky['XINTERP'][0],float);self.sy=np.asarray(sky['YINTERP'][0],float)
        self.key=f"{a['run']:06d}-{a['camcol']}-{a['field']:04d}"
        if self.pixel.shape!=self.bad.shape:raise ValueError('Official mask/frame grid mismatch')

    def stamp(self,x,y,radius=8,exclude_object=False):
        xx=int(round(float(x)));yy=int(round(float(y)));r=radius
        if xx-r<0 or yy-r<0 or xx+r>=self.pixel.shape[1] or yy+r>=self.pixel.shape[0]:return None
        gy,gx=np.mgrid[yy-r:yy+r+1,xx-r:xx+r+1];bad=self.bad[gy,gx]
        if np.any(bad[np.hypot(gx-x,gy-y)<=3.5]):return None
        ok=~bad&np.isfinite(self.pixel[gy,gx]);
        if exclude_object:ok&=~self.objects[gy,gx]
        gx,gy=gx[ok].astype(float),gy[ok].astype(float);data=self.pixel[gy.astype(int),gx.astype(int)];cal=self.calib[gx.astype(int)]
        sky=map_coordinates(self.sky,np.array([self.sy[gy.astype(int)],self.sx[gx.astype(int)]]),order=1,mode='nearest',prefilter=False)
        dn=data/cal+sky;variance=(np.maximum(dn,0)/self.a['gain']+self.a['dark_variance'])*cal**2
        good=np.isfinite(data)&np.isfinite(variance)&(variance>0)
        if good.sum()<100:return None
        return data[good],np.sqrt(variance[good]),gx[good],gy[good]

    def centroid(self,sky):
        initial=self.wcs.all_world2pix([sky],0)[0];stamp=self.stamp(*initial)
        if stamp is None:return None
        data,sigma,gx,gy=stamp
        def residual(point):return flux_profile(data,sigma,self.psf.sample(*point,gx,gy))[0]
        fit=least_squares(residual,initial,jac=lambda x:numerical_jacobian(residual,x,[.003,.003]),bounds=(initial-1.5,initial+1.5),max_nfev=100,ftol=1e-8,xtol=1e-8,gtol=1e-8)
        r,flux,bg,error=flux_profile(data,sigma,self.psf.sample(*fit.x,gx,gy));normal=fit.jac.T@fit.jac
        if not fit.success or np.linalg.matrix_rank(normal)<2:return None
        return dict(pixel=fit.x,published_minus_WCS_sky=None,centroid_covariance_pixel=np.linalg.inv(normal),flux=flux,flux_error=error,profile_SNR=flux/error,reduced_chi2=float(r@r/max(1,len(r)-4)),position_from_WCS=self.wcs.all_pix2world([fit.x],0)[0])

    def affine(self,center):
        point=self.wcs.all_world2pix([center],0)[0];sky=unproject([[100.,0.],[-100.,0.],[0.,100.],[0.,-100.]],center);p=self.wcs.all_world2pix(sky,0)
        plate=np.column_stack([(p[0]-p[1])/200,(p[2]-p[3])/200]);test=unproject([[1000,1000],[-1000,-1000]],center);actual=self.wcs.all_world2pix(test,0);model=point+np.array([[1000,1000],[-1000,-1000]])@plate.T
        if np.max(np.linalg.norm(actual-model,axis=1))>.001:raise ValueError('Local pixel/sky linearization fails over2.8arcsec')
        return point,plate


def calibrate(base,root):
    out=root/'pixelprep';out.mkdir(exist_ok=True);assets=read(root/'assets/private_field_assets.json');forty=read(base/'v2.2/ngc-early/private_frozen_forty.json')['targets'];allstars=[]
    for a in assets:
        with fits.open(a['catalogue'],memmap=False) as h:
            t=h[1].data;names={k.upper():k for k in t.names};mag=np.asarray(t['PSFMAG'][:,2]);err=np.asarray(t['PSFMAGERR'][:,2]);sky=np.column_stack([t['RA'],t['DEC']]);shape=np.asarray(t['OBJC_TYPE'] if 'OBJC_TYPE' in names else t['TYPE']);shape=shape[:,2] if shape.ndim==2 else shape
            good=np.isfinite(sky).all(axis=1)&(mag>16.5)&(mag<20)&(err>0)&(err<.05)&(shape==6)&(t['NCHILD']==0)&((t['RESOLVE_STATUS']&1)!=0)
            center=tuple(np.median(sky[good],axis=0));xy=project(*sky.T,center);tree=AngularTree(xy,center);d,_=tree.query(xy,k=2);good&=d[:,1]>=3000
            targets=AngularTree(project(*np.array([[r['ra'],r['dec']] for r in forty]).T,center),center);good&=targets.query(xy)[0]>1000
            chosen=sorted(np.flatnonzero(good),key=lambda i:sky_key(*sky[i]))[:160]
            for i in chosen:allstars.append(dict(field_tag=f"{a['run']:06d}-{a['camcol']}-{a['field']:04d}",position=sky[i],r=float(mag[i]),published_rowc=float(t['ROWC'][i,2]),published_sigma_mas=[1000*float(t['RAERR'][i]),1000*float(t['DECERR'][i])]))
    # Cross-frame calibrator identities and the validation split are fixed using
    # positions alone. No image-fit residual or target PM enters this split.
    points=np.array([r['position'] for r in allstars]);globalcenter=tuple(np.median(points,axis=0));tree=AngularTree(project(*points.T,globalcenter),globalcenter);neighbors=tree.query_ball_point(project(*points.T,globalcenter),700);parent=np.arange(len(points))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i,group in enumerate(neighbors):
        for j in group:
            x,y=find(i),find(j)
            if x!=y:parent[max(x,y)]=min(x,y)
    identities={i:sky_key(*points[find(i)]) for i in range(len(points))}
    for i,r in enumerate(allstars):r['physical_calibrator']=identities[i];r['validation']=int(identities[i][:8],16)%5==0
    save_json(out/'private_calibrator_position_registration.json',allstars)
    summaries=[];fitted=[]
    for a in assets:
        field=Field(base,a);target=out/field.key;target.mkdir(exist_ok=True);records=[]
        for r in [r for r in allstars if r['field_tag']==field.key]:
            fit=field.centroid(r['position'])
            if fit is None:continue
            # Numerical support/SNR defines whether a centroid was actually
            # measured; no clipping on position discrepancy to make a tie pass.
            if fit['profile_SNR']<20:continue
            records.append(dict(r,**fit))
        center=tuple(np.median([r['position'] for r in records],axis=0)) if records else (0.,0.)
        for r in records:r['published_minus_WCS_sky']=project(*np.array([r['position']]).T,center)[0]-project(*np.array([r['position_from_WCS']]).T,center)[0]
        train=[r for r in records if not r['validation']];test=[r for r in records if r['validation']];delta=np.median([r['published_minus_WCS_sky'] for r in train],axis=0) if train else np.zeros(2);res=np.array([r['published_minus_WCS_sky']-delta for r in test]);stat=dict(training_calibrators=len(train),independent_validation_calibrators=len(test),source_candidates=sum(r['field_tag']==field.key for r in allstars),measured_calibrators=len(records),constant_frame_offset_mas=delta)
        passed=False
        if len(train)>=15 and len(test)>=8:
            med=np.median(res,axis=0);nmad=1.4826*np.median(abs(res-med),axis=0);p68=float(np.quantile(np.linalg.norm(res,axis=1),.68));stat.update(validation_median_mas=med,validation_NMAD_mas=nmad,validation_radial_P68_mas=p68);passed=bool(np.max(abs(med))<=50 and p68<=100)
        # A physical calibrator receives the same bootstrap weight in every
        # field/run where it occurs. Existing C0 draws remain a separate term.
        draws=[]
        for b in range(48):
            weights=np.array([np.random.default_rng(int(sha256((r['physical_calibrator']+':'+str(b)).encode()).hexdigest()[:16],16)).exponential() for r in train]);values=np.array([r['published_minus_WCS_sky'] for r in train])
            if len(values):
                draw=[]
                for axis in range(2):
                    order=np.argsort(values[:,axis]);draw.append(values[order[np.searchsorted(np.cumsum(weights[order]),.5*weights.sum())],axis])
                draws.append(np.array(draw)-delta)
            else:draws.append(np.zeros(2))
        row_offset=float(np.median([r['published_rowc']-r['pixel'][1] for r in train])) if train else .5
        detail=dict(asset=a,field_tag=field.key,center_deg=center,constant_offset_mas=delta,calibration_draws=np.asarray(draws)-np.mean(draws,axis=0),rowc_minus_image_y=row_offset,tie_pass=passed,summary=stat)
        save_json(target/'private_frame_tie.json',detail);save_json(target/'private_calibrator_measurements.json',records)
        summaries.append(dict(field_tag=field.key,tie_pass=passed,**stat,metadata_rowc_minus_pixel_y=row_offset));print('V23_FRAME_TIE',field.key,len(train),len(test),passed,flush=True)
    summary=dict(fields=summaries,passed_physical_fields=sum(r['tie_pass'] for r in summaries),registered_physical_fields=len(assets),calibrator_registration_sha256=digest(out/'private_calibrator_position_registration.json'),slurm_job=os.environ['SLURM_JOB_ID'],scope='All13 frozen fields, positive flux/background profiled PSF centroids; split by shared physical calibration star before image fits. One constant native-field tie only, no higher-order fit. Validation targets not used to train offsets. Official per-pixel gain/dark/sky noise; shared calibrator bootstrap weights synchronized across frames and C0 separately retained. Frame-tie pass is not faint-centroid validation.')
    check_aggregate(summary);save_json(out/'v23_frame_registration_summary.json',summary)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('base',type=Path);p.add_argument('root',type=Path);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use Slurm')
    calibrate(a.base,a.root)
