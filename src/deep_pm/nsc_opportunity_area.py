"""Geometric calibrated joint opportunity area; never faint-source completeness."""
import argparse
from collections import Counter
import os
from pathlib import Path

from astropy.io import fits
import healpy as hp
from matplotlib.path import Path as Polygon
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .__main__ import tomllib
from .astrometry import project,years,sdss_tai_to_tt,utc_to_tt
from .continuous_reference import bind,read_offsets,verify_directory
from .continuous_summary import area
from .footprints import sdss_great_circle,sdss_window_mask,sdss_window_selection
from .io import digest,save_json
from .provenance import run_stage
from .public_summary import check_aggregate


def equal_area_grid(box,step):
    nx=int(np.ceil(np.ptp(box['ra'])/step));ny=int(np.ceil(np.ptp(box['dec'])/step))
    ra=np.linspace(*box['ra'],nx+1);sine=np.linspace(*np.sin(np.deg2rad(box['dec'])),ny+1)
    a,d=np.meshgrid((ra[:-1]+ra[1:])/2,np.rad2deg(np.arcsin((sine[:-1]+sine[1:])/2)))
    return np.c_[a.ravel(),d.ravel()],area(box)/(nx*ny)


def scan_members(table,sky,xy,center,margin):
    tree=cKDTree(xy);byrun={};epochs={}
    for row in table.itertuples(index=False):
        origin=project([row.RA],[row.DEC],center)[0]
        ix=np.asarray(tree.query_ball_point(origin,margin*3600000),int)
        if not len(ix):continue
        mu,nu=sdss_great_circle(sky[ix,0],sky[ix,1],row.NODE,row.INCL);width=(row.MU_END-row.MU_START)%360
        ix=ix[((mu-row.MU_START)%360<=width)&(nu>=min(row.NU_START,row.NU_END))&(nu<=max(row.NU_START,row.NU_END))]
        if len(ix):byrun.setdefault(int(row.RUN),[]).append(ix);epochs.setdefault(int(row.RUN),[]).append(float(row.epoch))
    return {k:np.unique(np.concatenate(v)) for k,v in byrun.items()},epochs


def estimate(chips,window,box,solution,step):
    sky,unit=equal_area_grid(box,step);center=tuple(float(np.mean(box[k])) for k in ('ra','dec'));xy=project(*sky.T,center);tree=cKDTree(xy);n=len(sky)
    layers={k:dict(count=np.zeros(n,int),first=np.full(n,np.inf),last=np.full(n,-np.inf)) for k in ('r','all')}
    def update(layer,indices,epoch):
        layer['count'][indices]+=1;layer['first'][indices]=np.minimum(layer['first'][indices],epoch);layer['last'][indices]=np.maximum(layer['last'][indices],epoch)
    for exposure,part in chips.groupby('exposure',sort=False):
        if 'NSC:'+str(exposure) not in solution:continue
        selected=[]
        for row in part.itertuples(index=False):
            corners=np.array([(getattr(row,'vertex_ra'+str(i)),getattr(row,'vertex_dec'+str(i))) for i in range(1,5)])
            if not np.isfinite(corners).all():continue
            corners=project(*corners.T,center);mid=corners.mean(axis=0);radius=np.linalg.norm(corners-mid,axis=1).max();ix=np.asarray(tree.query_ball_point(mid,radius+1),int)
            if len(ix):selected.append(ix[Polygon(np.vstack([corners,corners[0]]),closed=True).contains_points(xy[ix])])
        if not selected:continue
        ix=np.unique(np.concatenate(selected));epoch=float(years(utc_to_tt(float(np.median(part.mjd)))))
        update(layers['all'],ix,epoch)
        if set(part['filter'].astype(str))=={'r'}:update(layers['r'],ix,epoch)
    selected=window.loc[[f'SDSS:{int(r)}:{int(c)}' in solution for r,c in zip(window.RUN,window.CAMCOL)]]
    # Two independent field-center margins must recover the same exact native
    # scan-bound coverage; polygon/scan metadata do not imply source SNR.
    members,epochs=scan_members(selected,sky,xy,center,.30);other,_=scan_members(selected,sky,xy,center,.50)
    if set(members)!=set(other) or any(not np.array_equal(members[k],other[k]) for k in members):raise ValueError('SDSS native footprint changes with center margin')
    early=dict(count=np.zeros(n,int),first=np.full(n,np.inf),last=np.full(n,-np.inf))
    for run,ix in members.items():update(early,ix,float(np.median(epochs[run])))
    masks={};result={}
    for key,layer in layers.items():
        span=layer['last']-layer['first'];joint=np.maximum(layer['last'],early['last'])-np.minimum(layer['first'],early['first'])
        mask=(early['count']>=1)&(layer['count']>=2)&(span>=.60)&(joint>=5)
        masks[key]=mask;result[key]=dict(geometric_joint_opportunity_area_sqdeg=float(mask.sum()*unit),sdss_calibrated_covered_area_sqdeg=float((early['count']>0).sum()*unit),nsc_calibrated_covered_area_sqdeg=float((layer['count']>0).sum()*unit))
    pixel=hp.ang2pix(128,*sky.T,lonlat=True);spatial={str(k):dict(nside128_nominal_grid_area_sqdeg=float(np.count_nonzero(pixel==k)*unit),r_opportunity_area_sqdeg=float(np.count_nonzero((pixel==k)&masks['r'])*unit),all_nsc_opportunity_area_sqdeg=float(np.count_nonzero((pixel==k)&masks['all'])*unit)) for k in np.unique(pixel)}
    return dict(grid_points=n,equal_area_cell_sqdeg=unit,nominal_step_deg=step,layers=result,added_opportunity_area_sqdeg=float(np.count_nonzero(masks['all']&~masks['r'])*unit),nside128=spatial,sdss_margin_stability=True)


def run(work,preflight,shared,geometry,window_path):
    config=tomllib.loads(geometry.read_text());box=config['boxes']['c'];center=tuple(float(np.mean(box[k])) for k in ('ra','dec'));chips=pd.read_csv(preflight/'private_continuous_chip_opportunities.csv.gz');solution=read_offsets(shared/'private_c0.npz')
    with fits.open(window_path,memmap=True) as handle:
        t=handle[1].data;selection=sdss_window_mask(t,301)&sdss_window_selection(t,center,8.)
        columns=['RA','DEC','RUN','CAMCOL','NODE','INCL','MU_START','MU_END','NU_START','NU_END'];table=pd.DataFrame({k:np.asarray(t[k][selection],float) for k in columns});tai=np.asarray(t['TAI'][selection],float)
        if tai.ndim==2:tai=tai[:,2]
        table['epoch']=years(sdss_tai_to_tt(np.where((tai>=4e9)&(tai<6e9),tai,np.nan)));table=table.loc[np.isfinite(table.epoch)]
    runs={str(step):estimate(chips,table,box,solution,step) for step in (.05,.025)}
    result=dict(nominal_area_sqdeg=area(box),grids=runs,reference_sha256=digest(shared/'private_c0.npz'),chip_metadata_sha256=digest(preflight/'private_continuous_chip_opportunities.csv.gz'),sdss_window_sha256=digest(window_path),slurm_job=os.environ['SLURM_JOB_ID'],
        scope='Equal-solid-angle midpoint quadrature of real NSC CCD polygons and SDSS native scan boundaries, limited to physical C0-supported groups. At least one SDSS run, two NSC exposures, NSC span>=0.60yr and joint span>=5yr. Repeated CCDs in one exposure count once. Two grid resolutions give numerical sensitivity, not statistical confidence bounds. This is calibrated geometric opportunity area, NOT actual per-target SNR, successful extraction/association, usable faint-star area or certified-zero area; those require source/quality audits. No fixed NSC filter whitelist.')
    check_aggregate(result);save_json(work/'nsc_opportunity_area_summary.json',result)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('work','preflight','shared','geometry','window_path'):p.add_argument(n,type=Path)
    a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use a Slurm compute node')
    a.work.mkdir(parents=True,exist_ok=True);bind(a.work,'metadata_receipt.json',verify_directory(a.preflight,('private_continuous_chip_opportunities.csv.gz',)));bind(a.work,'reference_receipt.json',verify_directory(a.shared,('private_c0.npz',)));bind(a.work,'geometry.toml',a.geometry);bind(a.work,'window.fits',a.window_path)
    with run_stage(a.work,a.geometry,'C_opportunity_area','nsc-opportunity-area-v15',inputs=('metadata_receipt.json','reference_receipt.json','geometry.toml','window.fits'),outputs=('nsc_opportunity_area_summary.json',)) as go:
        if go:run(**{k:v for k,v in vars(a).items()})


if __name__=='__main__':main()
