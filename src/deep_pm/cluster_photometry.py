"""Freeze a deeper, spatially independent NGC 5466 photometric template."""
import argparse
import json
import os
from pathlib import Path

from astropy.io import fits
from astropy.wcs import WCS
import numpy as np

from .astrometry import project
from .blind_catalogue_quality import read
from .io import digest,save_json
from .ngc5466_demo import CENTER,TARGET,BACKGROUND,EDGES,COLORS,optical
from .public_summary import check_aggregate

TRAIN=(.23,.32)
TRAIN_BACKGROUND=(.48,float(np.sqrt(.48**2+.32**2-.23**2)))
MASK=1|2|4|8|16|32|64|512|2048|4096
DEPTH_R0=23.2
DEPTH_G0=23.8


def radius(sky):
    return np.degrees(np.arctan(np.linalg.norm(project(*np.asarray(sky).T,CENTER),axis=1)/206264806.24709636))


def pixel_quality(root,sky,transmission):
    sky=np.asarray(sky);n=len(sky)
    good=np.zeros(n,bool);mask_ok=np.zeros(n,bool);depth=np.full((n,2),np.nan);owners=np.zeros(n,int)
    with fits.open(root.parent/'survey-bricks.fits.gz',memmap=False) as handle:
        t=handle[1].data
        near=(abs(t['RA']-CENTER[0])<.85)&(abs(t['DEC']-CENTER[1])<.8)
        bricks=t[near].copy()
    missing=[];used=[]
    for brick in bricks:
        ix=np.flatnonzero((sky[:,0]>=brick['RA1'])&(sky[:,0]<brick['RA2'])&(sky[:,1]>=brick['DEC1'])&(sky[:,1]<brick['DEC2']))
        if not len(ix):continue
        owners[ix]+=1
        name=str(brick['BRICKNAME']).strip();folder=root/'coadd'/name[:3]/name
        kinds=('maskbits','nexp-g','nexp-r','depth-g','depth-r')
        paths={k:folder/f'legacysurvey-{name}-{k}.fits.fz' for k in kinds}
        if not all(p.exists() for p in paths.values()):missing.append(name);continue
        values={};native_shape=None
        for k,p in paths.items():
            with fits.open(p,memmap=False) as handle:
                image=handle[1];x,y=WCS(image.header).all_world2pix(sky[ix,0],sky[ix,1],0)
                x=np.floor(x+.5).astype(int);y=np.floor(y+.5).astype(int)
                inside=(x>=0)&(y>=0)&(x<image.shape[1])&(y<image.shape[0])
                if not inside.all():raise ValueError('Registered brick rectangle falls outside its pixel WCS')
                values[k]=np.asarray(image.data[y,x],float)
                if native_shape is None:native_shape=image.shape
                elif image.shape!=native_shape:raise ValueError('Photometric depth and mask image geometry differs')
        allowed=((values['maskbits'].astype(np.int64)&MASK)==0)&(values['nexp-g']>0)&(values['nexp-r']>0)
        mask_ok[ix]=allowed
        for axis,band in enumerate(('g','r')):
            inverse=values['depth-'+band]
            depth[ix,axis]=22.5-2.5*np.log10(np.where(inverse>0,5/np.sqrt(np.maximum(inverse,1e-99)),np.nan))+2.5*np.log10(transmission[axis])
        good[ix]=allowed&(depth[ix,0]>=DEPTH_G0)&(depth[ix,1]>=DEPTH_R0)
        used.append(name)
    if np.any(owners>1):raise ValueError('Pixel area assigned to multiple physical bricks')
    return good,mask_ok,depth,dict(used_bricks=used,missing_bricks=missing,unassigned_positions=int((owners==0).sum()))


def template(phot,areas):
    tr=phot.radius.between(*TRAIN).to_numpy();bg=phot.radius.between(*TRAIN_BACKGROUND).to_numpy()
    alpha=areas['train']/areas['training_background']
    bins=[]
    for lo,hi in zip(EDGES[:-1],EDGES[1:]):
        a=phot.color.to_numpy()[tr&phot.r0.between(lo,hi,inclusive='left').to_numpy()]
        b=phot.color.to_numpy()[bg&phot.r0.between(lo,hi,inclusive='left').to_numpy()]
        candidates=[]
        for color in np.arange(.03,1.,.05):
            on=int((abs(a-color)<.105).sum());off=int((abs(b-color)<.105).sum())
            excess=on-alpha*off;score=excess/np.sqrt(max(1,on+alpha**2*off))
            candidates.append((score,color,on,off,excess))
        score,color,on,off,excess=max(candidates)
        valid=excess>=5 and score>=2.5
        bins.append(dict(lower_r0=float(lo),upper_r0=float(hi),center_color=float(color) if valid else None,
            training_candidates=on,training_background_candidates=off,training_excess=float(excess),training_score=float(score)))
    return bins


def run(sweep,root,output):
    output.mkdir(parents=True,exist_ok=True)
    phot=optical(sweep)
    if phot.empty:raise ValueError('No existing optical parent sample')
    transmission=np.median(phot[['mw_transmission_g','mw_transmission_r']],axis=0)
    step=.002
    a=np.arange(CENTER[0]-.63+step/2,CENTER[0]+.63,step)
    d=np.arange(CENTER[1]-.56+step/2,CENTER[1]+.56,step)
    aa,dd=np.meshgrid(a,d);sky=np.column_stack([aa.ravel(),dd.ravel()]);rr=radius(sky)
    needed=rr<.535;sky=sky[needed];rr=rr[needed]
    good,mask,depth,coverage=pixel_quality(root,sky,transmission)
    weights=step**2*np.cos(np.deg2rad(sky[:,1]))
    rings=dict(train=TRAIN,training_background=TRAIN_BACKGROUND,target=TARGET,evaluation_background=BACKGROUND)
    areas={name:float(weights[good&(rr>=lo)&(rr<hi)].sum()) for name,(lo,hi) in rings.items()}
    mask_areas={name:float(weights[mask&(rr>=lo)&(rr<hi)].sum()) for name,(lo,hi) in rings.items()}
    if min(areas.values())<=0:raise ValueError('Depth-controlled photometric region has no pixel-area support')
    accepted,_,_,object_coverage=pixel_quality(root,phot[['ra','dec']].to_numpy(),transmission)
    phot=phot.loc[accepted].copy()
    bins=template(phot,areas)
    phot['cmd_v20']=False
    for b in bins:
        if b['center_color'] is not None:
            select=phot.r0.between(b['lower_r0'],b['upper_r0'],inclusive='left')
            phot.loc[select,'cmd_v20']=abs(phot.loc[select,'color']-b['center_color'])<.105
    registration=dict(photometry_sha256=digest(sweep),center_deg=CENTER,train_annulus_deg=TRAIN,
        training_background_annulus_deg=TRAIN_BACKGROUND,target_annulus_deg=TARGET,
        evaluation_background_annulus_deg=BACKGROUND,cmd_half_width=.105,template=bins,
        empirical_sequence=True,age_metallicity_distance='Not fitted or assumed in this spatially independent empirical sequence.',
        extinction='Original DR9 per-object MW transmission for colors/magnitudes; median g/r transmission for the fine pixel depth mask.',
        depth_r0_min=DEPTH_R0,depth_g0_min=DEPTH_G0,median_mw_transmission=transmission,
        area_quadrature_step_deg=step,effective_area_sqdeg=areas,mask_only_area_sqdeg=mask_areas,
        adopted_maskbits=MASK,mask_and_depth_source='Published DR9 south coadd maskbits, nexp-g/r and PSF depth maps; no source remeasurement.',
        scope='Train, training background, target and evaluation background are spatially disjoint. No motion or QSO result used to select CMD. Effective area is mask+depth grid quadrature, not image-level extraction completeness.')
    file=output/'sample_registration.json'
    if file.exists() and read(file)!=json.loads(json.dumps(registration,default=lambda v:v.tolist())):raise ValueError('Deep photometric registration changed')
    save_json(file,registration)
    phot.to_csv(output/'private_photometric_parent.csv.gz',index=False)
    cohorts={};hist={}
    for name,(lo,hi) in rings.items():
        use=phot.radius.between(lo,hi,inclusive='left')
        cohorts[name]={}
        for low in np.arange(18.,23.,.5):
            parent=use&phot.r0.between(low,low+.5,inclusive='left')
            cohorts[name][str(low)]=dict(photometric_parents=int(parent.sum()),cmd_candidates=int((parent&phot.cmd_v20).sum()))
        hist[name]=np.histogram2d(phot.loc[use,'r0'],phot.loc[use,'color'],bins=(EDGES,COLORS))[0]
    report=dict(registration_sha256=digest(file),selection=registration,cohorts=cohorts,
        optical_histograms=hist,r0_edges=EDGES,color_edges=COLORS,pixel_coverage=coverage,
        depth_mask_p16_p50_p84=np.quantile(depth[np.isfinite(depth).all(axis=1)],[.16,.5,.84],axis=0),
        slurm_job=os.environ['SLURM_JOB_ID'])
    check_aggregate(report);save_json(output/'v20_ngc_photometry_summary.json',report)
    print('DEEP_PHOTOMETRY_REGISTERED',areas,[(b['lower_r0'],b['center_color']) for b in bins],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('sweep','root','output'):p.add_argument(n,type=Path)
    a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use Slurm compute nodes')
    run(**vars(a))
