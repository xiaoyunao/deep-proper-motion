"""Frozen-coefficient stellar g mechanism diagnostics, independent of production.

Gaia residuals below are validation-only. Any future candidate training must use
same-star repeats or red/blue double differences without Gaia PM/position truth.
"""
import argparse
from collections import Counter,defaultdict
from hashlib import sha256
import os
from pathlib import Path
import re
from time import monotonic

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy import units as u
from astropy.time import Time
from scipy.spatial import cKDTree

from .__main__ import tomllib
from .astrometry import project
from .blind_catalogue_quality import read
from .continuous_reference import PRECISION,SPATIAL
from .coordinate_metric import forward_jacobian,sky_basis
from .dcr_geometry import zenith_vectors,tangent_geometry,BLANCO
from .io import save_json,digest
from .pilot_production import quality_detections
from .precision_v03 import calibration_groups
from .public_summary import check_aggregate
from .usability_cluster import summarize

COEFFICIENTS=(-28.32,-23.39)


def deep_neighbours(sky,root):
    """Published deep LS positions and g flux; partial NSC images, no truth claim."""
    frames=[];receipts={}
    for side in ('north','south'):
        for path in sorted((root/side/'sweep/9.0').glob('sweep-*.fits')):
            match=re.fullmatch(r'sweep-(\d{3})([pm])(\d{3})-(\d{3})([pm])(\d{3})\.fits',path.name)
            if not match:continue
            a,s,d,b,t,e=match.groups();lo=int(d)*(-1 if s=='m' else 1);hi=int(e)*(-1 if t=='m' else 1)
            overlap=(sky[:,0]>=int(a)-.005)&(sky[:,0]<=int(b)+.005)&(sky[:,1]>=lo-.005)&(sky[:,1]<=hi+.005)
            if not overlap.any():continue
            with fits.open(path,memmap=True) as h:
                table=h[1].data;ix=np.flatnonzero((table['RA']>=sky[:,0].min()-.005)&(table['RA']<=sky[:,0].max()+.005)&(table['DEC']>=sky[:,1].min()-.005)&(table['DEC']<=sky[:,1].max()+.005))
                names=['RA','DEC','FLUX_G','FLUX_R','TYPE','MASKBITS','FRACFLUX_G']
                frames.append(pd.DataFrame({k:np.asarray(table[k][ix]).astype(str if k=='TYPE' else float) for k in names if k in table.names}))
            receipts[str(path)]=digest(path)
    if not frames:return [dict(status='no_deep_catalogue') for _ in sky],receipts
    frame=pd.concat(frames,ignore_index=True).drop_duplicates(['RA','DEC']);unit=sky_basis(frame.RA,frame.DEC)[:,0,:];tree=cKDTree(unit)
    units=sky_basis(*sky.T)[:,0,:];distance,near=tree.query(units,k=2);distance=2*np.arcsin(np.clip(distance/2,0,1))*206264806.24709636
    result=[]
    for j,position in enumerate(sky):
        if distance[j,0]>1000 or distance[j,1]-distance[j,0]<300:
            result.append(dict(status='no_unique_deep_self'));continue
        own=int(near[j,0]);neighbors=tree.query_ball_point(units[j],2*np.sin(10000/(2*206264806.24709636)))
        neighbors=[k for k in neighbors if k!=own]
        ownflux=float(frame.FLUX_G.iloc[own]);r=dict(status='unique_deep_self',self_type=str(frame.TYPE.iloc[own]).strip(),self_flux_g=ownflux)
        if neighbors:
            xy=project(frame.RA.iloc[neighbors],frame.DEC.iloc[neighbors],position)
            k=int(np.argmin(np.linalg.norm(xy,axis=1)));idx=neighbors[k]
            r.update(neighbour_distance_mas=float(np.linalg.norm(xy[k])),neighbour_east_mas=float(xy[k,0]),neighbour_north_mas=float(xy[k,1]),
                neighbour_relative_g_flux=float(frame.FLUX_G.iloc[idx]/ownflux) if ownflux>0 else None)
        else:r.update(neighbour_distance_mas=10000.,neighbour_east_mas=0.,neighbour_north_mas=0.,neighbour_relative_g_flux=0.)
        result.append(r)
    return result,receipts


def finite_json(value):
    if isinstance(value,dict):return {k:finite_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple,np.ndarray)):return [finite_json(v) for v in value]
    if isinstance(value,(float,np.floating)) and not np.isfinite(value):return None
    return value


def summary(rows):
    if not rows:return dict(n=0)
    actual=np.asarray([r['native_residual_mas'] for r in rows]);h=np.asarray([r['native_geometry'] for r in rows]);color=np.array([r['color'] for r in rows])
    unit=h/np.linalg.norm(h,axis=1)[:,None];perp=np.column_stack([-unit[:,1],unit[:,0]])
    return dict(n=len(rows),color_p16_p50_p84=np.quantile(color,[.16,.5,.84]),
        g_mag_p16_p50_p84=np.quantile([r['band_mag'] for r in rows],[.16,.5,.84]),
        g_snr_p16_p50_p84=np.quantile([r['band_snr'] for r in rows],[.16,.5,.84]),
        raw_native_residual=summarize(actual),parallel_to_zenith_p16_p50_p84_mas=np.quantile(np.sum(actual*unit,axis=1),[.16,.5,.84]),
        perpendicular_to_zenith_p16_p50_p84_mas=np.quantile(np.sum(actual*perp,axis=1),[.16,.5,.84]),
        frozen_models={str(k):summarize(actual-k*(color-1.2)[:,None]*h) for k in COEFFICIENTS})


def run(root,lsroot,output):
    start=monotonic();output.mkdir(parents=True,exist_ok=True)
    input=root/'v2.1/stellar-attribution/private_stellar_positions.json.gz';p=read(input)
    plan=read(root/'v2.1/registration/private_coordinate_plan.json');entries=read(root/'v2.0/ngc_motion/private_cluster_diagnostics.json.gz')['entries']
    member={tuple(r['source_key']):r['row']['detection_ids'] for r in entries}
    candidates=[dict(m) for m in p['measurements'] if m['survey']=='NSC' and m['band']=='g' and m.get('color') is not None and np.isfinite(m['color'])]
    config=tomllib.loads(PRECISION.read_text());settings=tomllib.loads(SPATIAL.read_text())
    result=[];failures=Counter()
    for core in sorted({m['source_key'][0] for m in candidates}):
        entry=next(r for r in plan['selected'] if r['core']==core);source=Path(entry['source'])
        data,xy,time,sigma,keep=quality_detections(source,entry['graph_center'],config,settings['production'])
        keys=calibration_groups(data);lookup=pd.Index(data.detection_id)
        for m in [r for r in candidates if r['source_key'][0]==core]:
            ix=lookup.get_indexer(member[tuple(m['source_key'])]);ix=ix[ix>=0]
            ix=ix[(keys[ix]==m['calibration_group'])&data.band.iloc[ix].eq('g').to_numpy()&(abs(time[ix]-m['time'])<1e-8)]
            if len(ix)!=1:failures['non_unique_physical_measurement']+=1;continue
            det=data.iloc[ix[0]];magerr=float(det.magnitude_error)
            m.update(detection_id=str(det.detection_id),band_mag=float(det.magnitude),band_snr=1.0857362047581296/magerr if magerr>0 else np.nan,
                seeing=float(det.get('fwhm',np.nan)),detector_id=str(det.detector_id),x=float(det.get('x',np.nan)),y=float(det.get('y',np.nan)))
            result.append(m)
    # One deep-neighbour query per fixed source, not per repeated exposure.
    representatives={tuple(r['source_key']):r for r in result};ordered=sorted(representatives)
    sky=np.array([[representatives[k]['ra_deg'],representatives[k]['dec_deg']] for k in ordered])
    neighbours,receipts=deep_neighbours(sky,lsroot);deep=dict(zip(ordered,neighbours))
    measurements=[];unsupported=Counter()
    times=sorted({r['time'] for r in result if r['instrument']=='c4d'})
    stamps=Time(times,format='jyear',scale='tt');zz=zenith_vectors(stamps.utc.mjd)
    sidereal=stamps.sidereal_time('apparent',longitude=BLANCO[0]*u.deg).hour
    geometry_cache={t:(zz[j],float(sidereal[j]),stamps.utc.isot[j][:10]) for j,t in enumerate(times)}
    for m in result:
        m.update(deep[tuple(m['source_key'])]);m['deep_neighbour']=deep[tuple(m['source_key'])]
        if m['instrument']!='c4d':unsupported[m['instrument']]+=1;continue
        zenith,sidereal,night=geometry_cache[m['time']];zenith=np.asarray(zenith)[None]
        h,air=tangent_geometry(np.array([m['ra_deg']]),np.array([m['dec_deg']]),zenith,(m['ra_deg'],m['dec_deg']))
        a=forward_jacobian(m['ra_deg'],m['dec_deg'],next(r['graph_center'] for r in plan['selected'] if r['core']==m['source_key'][0]))
        residual=np.linalg.solve(a,np.asarray(m['c0_graph']))
        # Hour angle from real sidereal time and the independently recorded site.
        hour=float((sidereal-m['ra_deg']/15+12)%24-12)
        m.update(native_geometry=h[0],geometric_airmass=float(air[0]),hour_angle_hours=hour,
            utc_night=night,native_residual_mas=residual,site_geodetic_deg_m=list(BLANCO))
        if not np.isfinite(m['band_mag']+m['band_snr']):failures['invalid_actual_g_snr']+=1;continue
        measurements.append(m)
    save_json(output/'private_stellar_g_mechanism.json',finite_json(dict(measurements=measurements,unsupported_instrument_counts=unsupported,LS_file_receipts=receipts)))
    exposures=[]
    for name,part in pd.DataFrame(measurements).groupby('calibration_group',sort=True):
        rows=part.to_dict('records');h=np.asarray(part.native_geometry.tolist())
        exposures.append(dict(exposure=name,night=rows[0]['utc_night'],stars=len(rows),
            tangent_zenith_median=np.median(h,axis=0),hour_angle_median=float(part.hour_angle_hours.median()),
            geometric_airmass=float(part.geometric_airmass.median()),metadata_airmass=float(part.airmass.median()) if np.isfinite(part.airmass.median()) else None))
    # Public geometry per physical exposure is a declared observation group,
    # never a private source coordinate or target identity.
    nights=[]
    for night in sorted({r['night'] for r in exposures}):
        exp=[e for e in exposures if e['night']==night];names=[e['exposure'] for e in exp]
        identities=[{tuple(m['source_key']) for m in measurements if m['calibration_group']==name} for name in names]
        common=set.intersection(*identities) if len(identities)>1 else set()
        h=np.array([e['tangent_zenith_median'] for e in exp])
        nights.append(dict(night=night,exposures=len(exp),common_stars=len(common),
            hour_angle_span_hours=float(np.ptp([e['hour_angle_median'] for e in exp])),
            geometric_vector_span=float(np.max(np.linalg.norm(h[:,None]-h[None,:],axis=-1)))))
    # Fixed actual-g conditions are diagnostics; show effective colour support,
    # not a second fit of the historical coefficients to Gaia residuals.
    groups={}
    for low,high in [(5,10),(10,20),(20,40),(40,100),(100,100000)]:
        rows=[m for m in measurements if low<=m['band_snr']<high]
        for tag,fn in [('all',lambda m:True),('deep_isolated',lambda m:m['deep_neighbour'].get('neighbour_distance_mas',0)>=3000)]:
            groups[f'SNR{low}_{high}__{tag}']=summary([m for m in rows if fn(m)])
    report=dict(input_sha256=digest(input),fixed_g_measurements=len(candidates),physical_recovered=len(result),
        DECam_geometry_measurements=len(measurements),deep_source_support=dict(Counter(m['deep_neighbour']['status'] for m in result)),
        failures=dict(failures),unsupported_instrument_counts=dict(unsupported),exposure_geometry=exposures,night_geometry=nights,
        frozen_coefficients_mas_per_mag_tanz=list(COEFFICIENTS),color_pivot_g_minus_i=1.2,
        actual_g_snr_and_deep_neighbour_validation=groups,elapsed_seconds=monotonic()-start,slurm_job=os.environ['SLURM_JOB_ID'],
        scope='Validation-only frozen historical k, no refit to Gaia. Same original physical source members; actual g mag/SNR and deep LS neighbours. DECam uses Blanco; other instruments explicitly not assigned Blanco. LS partly shares imaging. Geometry predictor only, no full aberration/refraction reapplied. Independent-night and double-difference candidate requires separate identifiability check; main catalogue unchanged.')
    check_aggregate(report);save_json(output/'v22_stellar_g_geometry_summary.json',report)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','lsroot','output'):p.add_argument(name,type=Path)
    a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use Slurm')
    run(**vars(a))
