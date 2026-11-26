"""Actual CCD/scan opportunity with spherical searches and local native support."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import os
import json
from hashlib import sha256
from pathlib import Path

from astropy.io import fits
import healpy as hp
import numpy as np
import pandas as pd

from .astrometry import years,sdss_tai_to_tt
from .blind_catalogue_quality import read
from .continuous_reference import read_offsets,reference_frame,verify_directory
from .continuous_summary import area
from .coordinate_metric import forward_jacobian
from .exposure_catalogues import inside
from .footprints import sdss_window_mask,sdss_window_selection
from .io import digest,save_json
from .nsc_opportunity_area import equal_area_grid,opportunity_masks
from .public_summary import check_aggregate


def one(task):
    cores,component,footprint,root,window_path=task;root=Path(root);dest=root/'area'/f"{component}_{cores[0]['window']}";dest.mkdir(parents=True,exist_ok=True)
    metadata=read(root/'graph/v22_global_graph_summary.json')
    graphfiles={r['component']:root/'reference'/f"component-{r['component']:04d}"/'private_c0_unfiltered.npz' for r in metadata['components'] if r['status']=='ready'}
    frame=Path(footprint)/'private_transfer_chips.csv.gz'
    verify_directory(Path(footprint),('private_transfer_chips.csv.gz','transfer_footprint_summary.json'))
    if read(Path(footprint)/'transfer_footprint_summary.json')['chip_sha256']!=digest(frame):raise ValueError('Changed measured CCD footprints')
    ticket=dict(reference_sha256={str(c):digest(p) for c,p in graphfiles.items()},CCD_sha256=digest(frame),SDSS_window_sha256=digest(window_path),
        implementation_sha256=digest(Path(__file__)),footprint_kernel_sha256=digest(Path(__file__).with_name('nsc_opportunity_area.py')),
        core_geometry_sha256=sha256(json.dumps({e['core']:e['spec'] for e in cores},sort_keys=True).encode()).hexdigest(),steps=[.05,.025])
    summary_path=dest/'v22_native_opportunity_core_region_summary.json'
    if summary_path.exists():
        prior=read(summary_path)
        if prior['ticket']!=ticket:raise ValueError('Changed native opportunity area inputs')
        return prior
    solutions={c:read_offsets(p) for c,p in graphfiles.items()}
    centers={c:reference_frame(p) for c,p in graphfiles.items()}
    covariances={c:{g:np.cov(v[1],rowvar=False) for g,v in solution.items()} for c,solution in solutions.items()}
    domain=[e['spec'] for e in cores];box={key:[min(s[key][0] for s in domain),max(s[key][1] for s in domain)] for key in ('ra','dec')}
    field_center=[sum(box[k])/2 for k in ('ra','dec')]
    chips=pd.read_csv(frame,dtype={'exposure':str})
    with fits.open(window_path,memmap=True) as handle:
        table=handle[1].data;selected=sdss_window_mask(table,301)&sdss_window_selection(table,field_center,8.)
        columns=['RA','DEC','RUN','CAMCOL','NODE','INCL','MU_START','MU_END','NU_START','NU_END']
        scan=pd.DataFrame({k:np.array(table[k][selected],float,copy=True) for k in columns});tai=np.array(table['TAI'][selected],float,copy=True)
        if tai.ndim==2:tai=tai[:,2]
        scan['epoch']=years(sdss_tai_to_tt(np.where((tai>=4e9)&(tai<6e9),tai,np.nan)));scan=scan.loc[np.isfinite(scan.epoch)]
    grids={}
    for step in (.05,.025):
        sky,unit=equal_area_grid(box,step);covered=np.zeros(len(sky),np.uint8)
        for s in domain:covered+=inside(*sky.T,s)
        if np.any(covered>1):raise ValueError('Overlapping registered core ownership')
        sky=sky[covered==1]
        early=np.zeros(len(sky),bool);late={tag:np.zeros(len(sky),bool) for tag in ('r','all')};masks={tag:np.zeros(len(sky),bool) for tag in ('r','all')}
        component_support={}
        for c,solution in solutions.items():
            def accept(key,points):
                if key not in covariances[c]:return np.zeros(len(points),bool)
                b=np.linalg.inv(forward_jacobian(*points.T,centers[c]));cv=b@covariances[c][key]@b.transpose(0,2,1)
                return np.sqrt(np.maximum(cv[:,[0,1],[0,1]],0)).max(axis=1)<=12
            first,second,valid=opportunity_masks(chips,scan,sky,field_center,solution,calibration_accept=accept)
            early|=first
            for tag in ('r','all'):late[tag]|=second[tag];masks[tag]|=valid[tag]
            component_support[str(c)]=float(valid['all'].sum()*unit)
        # OR complete component opportunities; NEVER combine an early field
        # from one independent reference frame with late data from another.
        pixel=hp.ang2pix(128,*sky.T,lonlat=True)
        spatial={str(p):dict(nside128_nominal_grid_area_sqdeg=float(np.sum(pixel==p)*unit),
            r_opportunity_area_sqdeg=float(np.sum((pixel==p)&masks['r'])*unit),
            all_nsc_opportunity_area_sqdeg=float(np.sum((pixel==p)&masks['all'])*unit)) for p in np.unique(pixel)}
        grids[str(step)]=dict(grid_points=len(sky),equal_area_cell_sqdeg=unit,nominal_step_deg=step,independent_reference_component_opportunity_sqdeg=component_support,
            layers={tag:dict(geometric_joint_opportunity_area_sqdeg=float(mask.sum()*unit),sdss_calibrated_covered_area_sqdeg=float(early.sum()*unit),nsc_calibrated_covered_area_sqdeg=float(late[tag].sum()*unit)) for tag,mask in masks.items()},
            added_opportunity_area_sqdeg=float(np.sum(masks['all']&~masks['r'])*unit),nside128=spatial,sdss_margin_stability=True)
    report=dict(ticket=ticket,component=component,window=cores[0]['window'],nominal_area_sqdeg=sum(area(s) for s in domain),grids=grids,
        slurm_job=os.environ['SLURM_JOB_ID'],scope='Real CCD polygons and native great-circle SDSS run/camcol fields; spherical candidate radii, field-specific SDSS r times; full-reference native per-position12mas support. All complete reference components tested; compatible early/late opportunities ORed after component-specific time conditions, not mixed across independent gauges. At least2 NSC physical exposures, NSC span0.6yr and joint5yr. Static-sky geometric/time opportunity only, not faint extraction, target SNR, stellar identity or certified zero area. Published-WCS boundaries used; tiny C0 boundary shifts remain below grid resolution and are not treated as source astrometry.')
    check_aggregate(report);save_json(summary_path,report);print('NATIVE_OPPORTUNITY_REGION',component,report['window'],flush=True);return report


def run(root,window,workers):
    plan=read(root/'registration/private_full_coordinate_plan.json');production=read(root.parent/'v1.9/batch/science/private_continuation_manifest.json');mapping={}
    for p in production['partitions']:
        for c,w in p['region_by_core'].items():mapping[c]=str(Path(p['owned']).parent/w/'footprint')
    graph=read(root/'graph/v22_global_graph_summary.json');groups={}
    for e in plan['cores']:
        matches=[r['component'] for r in graph['components'] if r['status']=='ready' and np.allclose(e['center'],r['center_deg'],atol=1e-12,rtol=0)]
        if len(matches)!=1:raise ValueError('Area component requires explicit connected-domain mapping')
        groups.setdefault((matches[0],mapping[e['core']]),[]).append(e)
    tasks=[(cores,c,footprint,str(root),window) for (c,footprint),cores in groups.items()]
    with ProcessPoolExecutor(max_workers=workers) as pool:results=list(pool.map(one,tasks))
    grids={}
    # Different region grids have different exact cell areas; sum their
    # already measured areas, never pretend they share an identical quadrature.
    for step in ('0.05','0.025'):
        by_pixel={};layers={};added=0.
        for r in results:
            g=r['grids'][step];added+=g['added_opportunity_area_sqdeg']
            for tag,values in g['layers'].items():
                target=layers.setdefault(tag,{})
                for k,v in values.items():target[k]=target.get(k,0)+v
            for p,values in g['nside128'].items():
                target=by_pixel.setdefault(p,{})
                for k,v in values.items():target[k]=target.get(k,0)+v
        grids[step]=dict(layers=layers,added_opportunity_area_sqdeg=added,nside128=by_pixel)
    result=dict(nominal_registered_area_sqdeg=sum(r['nominal_area_sqdeg'] for r in results),regions=len(results),grids=grids,
        all_band_resolution_difference_sqdeg=abs(grids['0.025']['layers']['all']['geometric_joint_opportunity_area_sqdeg']-grids['0.05']['layers']['all']['geometric_joint_opportunity_area_sqdeg']),
        slurm_job=os.environ['SLURM_JOB_ID'],scope='Sum over nonoverlapping frozen ownership cores and physical reference components. Two equal-area quadratures expose resolution sensitivity. Native support/time opportunity, occupied source pixels and scientifically certified area are distinct. No per-source raw rows in this public map.')
    check_aggregate(result);save_json(root/'area/v22_native_opportunity_area_summary.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('window',type=Path);p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use Slurm')
    run(a.root,a.window,a.workers)
