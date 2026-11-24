"""NGC 5466 outskirts: independent photometric selection and frozen motions."""

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path

from astropy.io import fits
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .__main__ import tomllib
from .astrometry import project, years
from .blind_catalogue_quality import read
from .continuous_reference import read_offsets, reference_frame, verify_directory
from .continuous_summary import estimate
from .dark_precision import query_unique
from .exposure_catalogues import inside
from .io import digest, save_json
from .nsc_ownership_overlay import owned_records
from .precision_v03 import calibration_groups
from .public_summary import check_aggregate

CENTER = (211.3637, 28.5344)
TRAIN = (.025, .070)
TARGET = (.075, .210)
BACKGROUND = (.400, float(np.sqrt(.4**2+.210**2-.075**2)))
EDGES = np.arange(18.,23.01,.5)
COLORS = np.arange(-.2,1.61,.05)
SEED = 5466


def optical(path):
    with fits.open(path,memmap=True) as f:
        t=f[1].data
        a,d=t['RA'],t['DEC']
        sky=(abs(a-CENTER[0])<.65)&(abs(d-CENTER[1])<.58)
        columns=('ra','dec','type','flux_g','flux_r','flux_ivar_g','flux_ivar_r',
                 'mw_transmission_g','mw_transmission_r','maskbits')
        p=pd.DataFrame({c:np.asarray(t[c.upper()][sky]).astype(str if c=='type' else float) for c in columns})
    finite=(p.flux_g>0)&(p.flux_r>0)&(p.mw_transmission_g>0)&(p.mw_transmission_r>0)
    unmasked=(p.maskbits.astype('int64') & (1|2|4|8|16|32|64|512|2048|4096))==0
    ok=finite & p.type.str.strip().eq('PSF') & unmasked
    for band in ('g','r'):
        ok &= p['flux_'+band]*np.sqrt(p['flux_ivar_'+band].clip(lower=0))>=5.
    p=p.loc[ok].copy()
    p['r0']=22.5-2.5*np.log10(p.flux_r/p.mw_transmission_r)
    p['color']=-2.5*np.log10((p.flux_g/p.mw_transmission_g)/(p.flux_r/p.mw_transmission_r))
    p['radius']=np.degrees(np.arctan(np.linalg.norm(project(p.ra,p.dec,CENTER),axis=1)/206264806.24709636))
    return p


def register(photometry, output):
    train=photometry.loc[photometry.radius.between(*TRAIN)]
    background=photometry.loc[photometry.radius.between(*BACKGROUND)]
    a=np.diff(np.square(TRAIN))[0]/np.diff(np.square(BACKGROUND))[0]
    bins=[]
    controls=np.arange(.03,1.,.05)
    for lo,hi in zip(EDGES[:-1],EDGES[1:]):
        tr=train.loc[train.r0.between(lo,hi,inclusive='left'),'color'].to_numpy()
        bg=background.loc[background.r0.between(lo,hi,inclusive='left'),'color'].to_numpy()
        result=[]
        for c in controls:
            m=np.abs(tr-c)<.105
            on=int(m.sum())
            off=int(np.count_nonzero(abs(bg-c)<.105))
            excess=on-a*off
            result.append((excess/np.sqrt(max(1,on+a*a*off)),c,on,off,excess))
        score,ridge,on,off,excess=max(result)
        valid=excess>=5 and score>=2.5
        bins.append(dict(lower_r0=lo,upper_r0=hi,center_color=ridge if valid else None,
                         train_in_strip=on,control_in_strip=off,excess=excess))
    photometry=photometry.copy()
    photometry['cmd']=False
    for b in bins:
        if b['center_color'] is not None:
            mask=photometry.r0.between(b['lower_r0'],b['upper_r0'],inclusive='left')
            photometry.loc[mask,'cmd']=abs(photometry.loc[mask,'color']-b['center_color'])<.105
    registration=dict(external_photometry_sha256=photometry.attrs['photometry_sha256'],center_deg=CENTER,
                      photometric_selection='Legacy Surveys DR9 south PSF, SNR≥5 in g and r, dereddened, optical masks',
                      train_annulus_deg=TRAIN,target_annulus_deg=TARGET,background_annulus_deg=BACKGROUND,
                      independently_fitted_cmd=bins,cmd_half_width_mag=.105,source_matching_max_arcsec=.7,
                      source_matching_second_neighbor_margin_arcsec=.3)
    file=output/'sample_registration.json'
    if file.is_file() and read(file)!=json.loads(json.dumps(registration,default=float)):
        raise ValueError('Photometric sample registration changed')
    save_json(file,registration)
    return photometry,registration


def footprints(frozen):
    summary = frozen/'continuation_production_science_summary.json'
    receipt = read(frozen/'continuation_science_receipt.json')
    if digest(summary)!=receipt['combined_science_sha256']:
        raise ValueError('Frozen science scope changed')
    for partition in read(frozen/'private_continuation_manifest.json')['partitions']:
        owned=Path(partition['owned'])
        specs=tomllib.loads(Path(partition['geometry']).read_text())['fields']
        relevant=[]
        for core in partition['region_by_core']:
            box=specs[core]
            if abs(np.mean(box['ra'])-CENTER[0])<1.6 and abs(np.mean(box['dec'])-CENTER[1])<1.3:
                relevant.append(core)
        if relevant:
            verify_directory(owned,('continuous_science_summary.json','fits.json','private_owner_overlay.json'))
            if digest(owned/'continuous_science_summary.json')!=receipt['graphs'][partition['label']]['science_receipt_sha256']:
                raise ValueError('Frozen owner solution changed')
            yield owned, specs, relevant


def pair_photometry(phot, records):
    positions=np.array([r[2]['position_deg'] for r in records])
    found, nearest=query_unique(cKDTree(project(*positions.T,CENTER)),project(phot.ra,phot.dec,CENTER),700,300)
    return {int(i):records[nearest[i]] for i in np.flatnonzero(found)}


def source_covariances(rows, original_fits):
    by_core = defaultdict(list)
    for origin,_,r in rows:
        by_core[origin].append(r)
    values=[]
    for origin, records in by_core.items():
        fit=Path(original_fits[origin])
        graph=(fit/'reference_receipt.json').resolve().parent/'private_c0.npz'
        offsets=read_offsets(graph)
        graph_hash=digest(graph)
        source=(fit/'source_receipt.json').resolve().parent
        verify_directory(source,('detections.csv',))
        columns=['detection_id','survey','detector_id','exposure_id','observation_time','sigma_east_mas','sigma_north_mas']
        measurements=pd.read_csv(source/'detections.csv',usecols=columns)
        wanted={i for r in records for i in r['detection_ids']}
        measurements=measurements.loc[measurements.detection_id.isin(wanted)].set_index('detection_id',drop=False)
        legacy=pd.read_csv(fit/'legacy.csv',dtype={'objID':str})
        legacy=legacy.loc[legacy.PM_SOURCE.str.strip().eq('DESI')].drop_duplicates('objID',keep=False).set_index('objID')
        legacy_ids=set(legacy.index)
        for row in records:
            if row['reference_model_sha256']!=graph_hash:
                raise ValueError('Changed reference input')
            d=measurements.loc[row['detection_ids']]
            keys=calibration_groups(d.reset_index(drop=True))
            t=years(d.observation_time.to_numpy(float))
            sigma=np.hypot(d[['sigma_east_mas','sigma_north_mas']].to_numpy(float),50.)
            w=np.asarray(row['robust_weight'])[:,None]/sigma**2
            design=np.column_stack((np.ones(len(t)),t-row['reference_epoch']))
            independent=np.asarray(row['independent_tangent_covariance'])
            stacked=np.stack([offsets[k][1] for k in keys],axis=1)
            shifts=np.concatenate([stacked[:,:,i]@(independent[2*i:2*i+2,2*i:2*i+2]@(design.T*w[:,i])).T
                                   for i in range(2)],axis=1)
            jac=np.asarray(row['state_jacobian'])
            np.testing.assert_allclose(np.cov(shifts,rowvar=False),row['reference_tangent_covariance'],atol=2.e-7,rtol=2.e-8)
            shifted_icrs=shifts@jac[[1,3],:].T
            independent_icrs=(jac@independent@jac.T)[np.ix_([1,3],[1,3])]
            xiao_ids={i.partition(':')[2] for i in row['detection_ids'] if i.startswith('SDSS:')} & legacy_ids
            xiao=None
            if len(xiao_ids)==1:
                a=legacy.loc[next(iter(xiao_ids))]
                xp=a[['PMRA','PMDEC','PMRA_ERR','PMDEC_ERR']].to_numpy(float)
                if np.isfinite(xp).all() and np.all(xp[2:]>0) and np.all(abs(xp[:2])<1e5):
                    xiao=xp
            values.append((row,independent_icrs,shifted_icrs,graph_hash,xiao))
    return values


def estimate_mean(rows, qso):
    out=dict(n=len(rows))
    if len(rows)<5:
        return out
    pm=np.asarray([r[0]['pm_icrs_masyr'] for r in rows])
    errors=np.asarray([r[0]['pm_error_masyr'] for r in rows])
    w=1/np.mean(errors**2,axis=1)
    w/=w.sum()
    independent=sum(a*a*r[1] for a,r in zip(w,rows))
    domains=defaultdict(list)
    for a,r in zip(w,rows):
        domains[r[3]].append(a*r[2])
    common=sum(np.cov(np.sum(shifts,axis=0),rowvar=False) for shifts in domains.values())
    rng=np.random.default_rng(SEED)
    ids=rng.integers(0,len(rows),(256,len(rows)))
    object_reweight=np.sum(pm[ids]*w[ids,None],axis=1)/np.sum(w[ids],axis=1)[:,None]
    out.update(weighted_icrs_mean=w@pm, median_icrs=np.median(pm,axis=0),
               independent_measurement_covariance=independent,
               synchronous_reference_covariance=common,
               object_resample_95=np.quantile(object_reweight,[.025,.975],axis=0).T,
               observed_axis_p16_p50_p84=np.quantile(pm,[.16,.5,.84],axis=0),
               reference_domains=len(domains),local_qso_zero_uncertainty=qso)
    return out


def run(frozen, sweep, output):
    output.mkdir(parents=True,exist_ok=True)
    phot=optical(sweep)
    if phot.empty:
        raise ValueError('The photometry tile has zero qualifying detections in the target footprint')
    phot.attrs['photometry_sha256']=digest(sweep)
    phot, registration=register(phot,output)
    target=phot.radius.between(*TARGET)
    background=phot.radius.between(*BACKGROUND)
    records=[]
    qso=[]
    fits_by_core={}
    coverage=0.
    geometry=[]
    for owned, specs, cores in footprints(frozen):
        manifest=read(owned/'fits.json')
        for core in cores:
            box=specs[core]
            geometry.append(box)
            coverage+=np.degrees(np.ptp(np.sin(np.deg2rad(box['dec']))))*np.ptp(box['ra'])
            owned_rows=list(owned_records(owned,core))
            kept={(origin,slot) for origin,slot,row in owned_rows
                  if np.linalg.norm(project(*row['position_deg'],CENTER)[0]) <= 1.15*3600000}
            records.extend((origin,slot,row) for origin,slot,row in owned_rows if (origin,slot) in kept)
            for origin in {c for c,_ in kept}:
                fits_by_core[origin]=manifest[origin]
            folder=Path(manifest[core])
            verify_directory(folder,('private_blind_qso.json',))
            for b in read(folder/'private_blind_qso.json'):
                if (core,b['source_slot']) in kept and b['magnitude'] is not None:
                    qso.append(dict(b,motion={'C0':b['values']['blind_all'][0]}))
    if not records:
        raise ValueError('Frozen selected area has no catalogue output')
    exists=np.logical_or.reduce([inside(phot.ra,phot.dec,box) for box in geometry])
    if np.any((target|background)&~exists):
        raise ValueError('Footprint cuts a target or background sample')
    matched=pair_photometry(phot,records)
    by_group={}
    qso_quality=estimate(qso,('C0',),SEED)
    qso_zero=qso_quality.get('methods',{}).get('C0',{})
    r0=phot.r0.to_numpy()
    selected=phot.cmd.to_numpy()
    optical_summary={}
    for name,mask in [('target',target.to_numpy()),('background', background.to_numpy())]:
        by_group[name]={}
        catalogued = {i:matched[i] for i in np.flatnonzero(mask&selected&(r0>=18)&(r0<23)) if i in matched}
        evaluated = source_covariances(list(catalogued.values()),fits_by_core)
        row_results = {id(v[0]):v for v in evaluated}
        for label,mag in [('all',(r0>=18)&(r0<23)),('bright',(r0>=18)&(r0<21)),('faint',(r0>=21)&(r0<23))]:
            relevant=mask&mag
            candidates=relevant&selected
            source=[catalogued[i] for i in np.flatnonzero(candidates) if i in catalogued]
            values=[row_results[id(v[2])] for v in source]
            cross=[v for v in values if v[-1] is not None]
            compar=dict(n=len(cross))
            if len(cross)>=5:
                delta=np.array([np.asarray(v[0]['pm_icrs_masyr'])-v[-1][:2] for v in cross])
                compar.update(median_difference=np.median(delta,axis=0),
                              xiao_median=np.median([v[-1][:2] for v in cross],axis=0),
                              allband_median=np.median([v[0]['pm_icrs_masyr'] for v in cross],axis=0))
            by_group[name][label]=dict(
                photometric_parents=int(relevant.sum()),cmd_candidates=int(candidates.sum()),
                matched_frozen_motion=len(source),paired_xiao=compar,
                not_linked_to_xiao_by_unique_sdss_member=len(values)-len(cross),
                candidate_motion=estimate_mean(values,qso_zero),
                observed_motion_hist=np.histogram2d(*np.array([v[0]['pm_icrs_masyr'] for v in values]).T,
                    bins=np.arange(-20,20.01,2))[0] if values else None)
        optical_summary[name]=np.histogram2d(r0[mask],phot.color.to_numpy()[mask],bins=(EDGES,COLORS))[0]
    delta = project(phot.ra,phot.dec,CENTER)/3600000
    grid=np.linspace(-.5,.5,25)
    spatial=np.histogram2d(*delta[phot.cmd].T,bins=(grid,grid))[0].astype(int)
    # Release count maps rather than star coordinates.
    spatial=np.where(spatial>=3,spatial,0)
    result=dict(spatial_cmd_counts=dict(tile_counts=spatial,edges_deg=grid,display_minimum=3),scope='DR9 optical CMD selected before opening the motion files, using inner photometry distinct from the target/outskirts sample. Weighted candidate motions include field contamination. A missing direct Xiao SDSS member match does not independently establish a new star or a catalogue absence.',
                sample_registration_sha256=digest(output/'sample_registration.json'),selection=registration,
                nominal_input_core_area_sqdeg=coverage,input_cores=len(geometry), cohorts=by_group,
                local_blind_qso=qso_quality, local_blind_qso_radius_deg=1.15,
                optical_cmd=dict(histograms=optical_summary,r_edges=EDGES,color_edges=COLORS),
                slurm_job=os.environ['SLURM_JOB_ID'])
    check_aggregate(result)
    save_json(output/'v19_ngc5466_summary.json',result)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('frozen','sweep','output'):
        parser.add_argument(name,type=Path)
    args=parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        parser.error('Use a Slurm compute node')
    run(**vars(args))
