"""Fixed stellar identities: actual-epoch positions, C0 response and colour tests.

Gaia positions and space motion are validation labels only. Nothing in this
module discovers tracks or changes source membership or production calibration.
"""
import argparse
from collections import Counter,defaultdict
import gzip
import json
import os
from pathlib import Path
from time import monotonic

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from astropy.coordinates import get_body_barycentric
from astropy.time import Time

from . import access
from .__main__ import tomllib
from .astrometry import apparent_positions,project,MAS_PER_RADIAN
from .blind_catalogue_quality import read
from .continuous_reference import PRECISION,SPATIAL,read_offsets
from .coordinate_metric import native_covariance,projected_covariance,forward_jacobian,fit_vector_motion,AngularTree
from .io import digest,save_json,json_default
from .pilot_production import quality_detections
from .precision_v03 import calibration_groups
from .public_summary import check_aggregate
from .standard_motion import icrs_motion
from .usability_cluster import summarize
from .usability_quality import split_identity

PARAMS=['ra','dec','parallax','pmra','pmdec']
CORR=['ra_dec','ra_parallax','ra_pmra','ra_pmdec','dec_parallax','dec_pmra','dec_pmdec','parallax_pmra','parallax_pmdec','pmra_pmdec']


def gaia_full(cache,cat,ids):
    selected=cat.loc[cat.source_id.isin(ids)].copy()
    fields=['source_id','ra','dec','ref_epoch','pmra','pmdec','parallax','radial_velocity']+[p+'_error' for p in PARAMS]+[p+'_corr' for p in CORR]
    if set(fields)<=set(selected):return selected,dict(fixed_id_queries=0,full_covariance_available=True)
    chunks=[];failures=0
    ordered=sorted(ids)
    for start in range(0,len(ordered),400):
        group=ordered[start:start+400]
        try:
            f=cache.checked('nsc',','.join(fields),'gaia_dr3.gaia_source','source_id IN ('+','.join(group)+')',cap=500)
            f.source_id=f.source_id.astype(str);chunks.append(f)
        except (RuntimeError,ValueError):failures+=1
    if chunks:
        extra=pd.concat(chunks,ignore_index=True).drop_duplicates('source_id').set_index('source_id')
        selected=selected.set_index('source_id');selected.update(extra)
        for key in extra:
            if key not in selected:selected[key]=extra[key]
        selected=selected.reset_index()
    return selected,dict(fixed_id_queries=(len(ordered)+399)//400,failed_queries=failures,
        full_covariance_available=set(fields)<=set(selected))


def propagated_gaia(row,dt,observer,center):
    pi=float(row.parallax);negative=pi<0;pi=max(0,pi)
    rv=float(row.radial_velocity) if pd.notna(row.radial_velocity) else 0.
    args=[float(row.ra),float(row.dec),float(row.pmra),float(row.pmdec),pi,rv]
    p=apparent_positions(*args,dt,observer,center)
    no_parallax=apparent_positions(*args,dt,np.zeros_like(observer),center)
    native_cov=None
    fields=[p+'_error' for p in PARAMS]+[p+'_corr' for p in CORR]
    if all(k in row and pd.notna(row[k]) for k in fields):
        error=np.array([row[k+'_error'] for k in PARAMS],float);cov=np.diag(error**2)
        for key in CORR:
            first,second=key.split('_');i,j=PARAMS.index(first),PARAMS.index(second)
            cov[i,j]=cov[j,i]=row[key+'_corr']*error[i]*error[j]
        derivatives=[]
        for k,step in enumerate([.1,.1,.1,.01,.01]):
            variations=[]
            for sign in (1,-1):
                a=args.copy()
                if k==0:a[0]+=sign*step/(3600000*np.cos(np.deg2rad(a[1])))
                elif k==1:a[1]+=sign*step/3600000
                elif k==2:a[4]+=sign*step
                else:a[k-1]+=sign*step
                variations.append(apparent_positions(*a,dt,observer,center))
            derivatives.append((variations[0]-variations[1])/(2*step))
        f=np.stack(derivatives,axis=-1);native_cov=f@cov@np.swapaxes(f,-1,-2)
    return p,no_parallax,native_cov,negative,pd.isna(row.radial_velocity)


def colour_pairs(frame):
    """Unique red/blue metadata matches in each physical detector/field group."""
    pairs=[];groups=0
    for name,g in frame.groupby('physical_group',sort=True):
        blue=g.loc[g.color.lt(.5)&g.crowding_mas.ge(2000)];red=g.loc[g.color.ge(2)&g.crowding_mas.ge(2000)]
        if blue.empty or red.empty:continue
        b=blue.reset_index(drop=True);r=red.reset_index(drop=True)
        # Metadata metric only: shared position, brightness, and crowding support.
        tree=cKDTree(np.column_stack([b.sky_e/180000,b.sky_n/180000,b.rmag/.5,np.log10(b.crowding_mas)/.3]))
        dist,near=tree.query(np.column_stack([r.sky_e/180000,r.sky_n/180000,r.rmag/.5,np.log10(r.crowding_mas)/.3]))
        taken=set();added=0
        for j in np.argsort(dist,kind='stable'):
            k=int(near[j])
            if k in taken:continue
            a,c=b.iloc[k],r.iloc[j]
            if np.hypot(a.sky_e-c.sky_e,a.sky_n-c.sky_n)>180000 or abs(a.rmag-c.rmag)>.5 or abs(np.log10(a.crowding_mas/c.crowding_mas))>.3:continue
            taken.add(k);added+=1
            pairs.append(dict(physical_group=name,survey=str(c.survey),band=str(c.band),split=int(c.split),
                delta_color=float(c.color-a.color),delta_ri=float(c.ri-a.ri) if np.isfinite(c.ri) and np.isfinite(a.ri) else None,
                raw_difference=np.array([c.raw_e-a.raw_e,c.raw_n-a.raw_n]),
                c0_difference=np.array([c.c0_e-a.c0_e,c.c0_n-a.c0_n]),
                airmass=float(np.nanmean([c.airmass,a.airmass])) if np.isfinite([c.airmass,a.airmass]).any() else None))
        groups+=added>0
    return pairs,groups


def run(registration,old_cluster,output,cache):
    start=monotonic();output.mkdir(parents=True,exist_ok=True)
    plan=read(registration);frozen=read(old_cluster);entries=frozen['entries']
    bright=[r for r in entries if r.get('gaia')];wanted={tuple(r['source_key']) for r in bright}
    old_keys={tuple(k) for name,keys in frozen['selected'].items() if name.startswith('old__target__') for k in keys}
    config=tomllib.loads(PRECISION.read_text());settings=tomllib.loads(SPATIAL.read_text())
    allmeasure=[];states=[];queries=[];missing=Counter();maximum_k=0.;maximum_state=0.
    for row in plan['selected']:
        chosen=[r for r in entries if r['source_key'][0]==row['core'] and (tuple(r['source_key']) in wanted or tuple(r['source_key']) in old_keys)]
        if not chosen:continue
        source=Path(row['source']);center=row['graph_center'];solution=read_offsets(Path(row['graph']))
        data,xy,t,sigma,keep=quality_detections(source,center,config,settings['production']);lookup=pd.Index(data.detection_id);keys=calibration_groups(data)
        cp=projected_covariance(data.ra_deg,data.dec_deg,center,native_covariance(sigma))
        cat=pd.read_csv(source/'validation/gaia.csv',dtype={'source_id':str})
        ids={str(r['gaia']['source_id']) for r in chosen if r.get('gaia')}
        cat,query=gaia_full(access.Cache(cache),cat,ids);queries.append(query);cat=cat.drop_duplicates('source_id').set_index('source_id')
        times=Time(t,format='jyear',scale='tt');unique,inverse=np.unique(t,return_inverse=True)
        earth=get_body_barycentric('earth',Time(unique,format='jyear',scale='tt')).xyz.to_value('au').T[inverse]
        rawfile=source/'sdss.csv';phot=pd.read_csv(rawfile,dtype=str).rename(columns=str.lower)
        if 'objid' in phot:phot.index='SDSS:'+phot.objid.astype(str)
        else:phot=phot.iloc[:0]
        for col in ('psfmag_g','psfmag_r','psfmag_i'):phot[col]=pd.to_numeric(phot.get(col),errors='coerce')
        phot=phot.loc[~phot.index.duplicated()]
        allsource=read(Path(row['fit'])/'private_standard_motion.json.gz')
        points=project(*np.asarray([r['position_deg'] for r in allsource]).T,center)
        neighbor=AngularTree(points,center).query(points,k=2)[0][:,1]
        for entry in chosen:
            old=entry['row'];ix=lookup.get_indexer(old['detection_ids'])
            if (ix<0).any():missing['missing_physical_members']+=1;continue
            z=np.asarray([solution[g][0] for g in keys[ix]])
            model=fit_vector_motion(xy[ix]-z,t[ix],cp[ix],old['reference_epoch'])
            raw=fit_vector_motion(xy[ix],t[ix],cp[ix],old['reference_epoch'],frozen_weight=model.weight)
            a=icrs_motion(model.coeff[0],model.coeff[1],center);b=icrs_motion(raw.coeff[0],raw.coeff[1],center)
            k=np.einsum('ap,pni->ani',a['state_jacobian'][[1,3]],model.influence)
            shift=-np.einsum('ani,ni->a',k,z);delta=a['pm_icrs_masyr']-b['pm_icrs_masyr']
            maximum_k=max(maximum_k,float(np.max(abs(delta-shift))))
            physical=tuple(entry['source_key']);early=data.survey.iloc[ix].eq('SDSS').to_numpy()
            gi=entry.get('sdss_g_minus_i');pmag=phot.reindex(data.detection_id.iloc[ix[early]])
            ri=float(np.nanmedian(pmag.psfmag_r-pmag.psfmag_i)) if len(pmag) else np.nan
            rmag=float(np.nanmedian(pmag.psfmag_r)) if len(pmag) else float(old['magnitude'])
            state=dict(source_key=physical,raw_motion=b['pm_icrs_masyr'],c0_motion=a['pm_icrs_masyr'],c0_predicted_change=shift,
                color=gi,ri=ri if np.isfinite(ri) else None,rmag=rmag,old_target=physical in old_keys,bright=entry.get('gaia') is not None,
                reference_state_draws=entry['reference_draws'],graph_sha256=row['graph_sha256'] if 'graph_sha256' in row else digest(Path(row['graph'])))
            if not entry.get('gaia'):states.append(state);continue
            identity=str(entry['gaia']['source_id'])
            if identity not in cat.index:missing['missing_gaia_label']+=1;states.append(state);continue
            label=cat.loc[identity];dt=times[ix].tcb.jyear-float(label.ref_epoch)
            prediction,no_pi,gcov,negative,no_rv=propagated_gaia(label,dt,earth[ix],center)
            state.update(gaia_motion=[float(label.pmra),float(label.pmdec)],
                raw_residual=b['pm_icrs_masyr']-[label.pmra,label.pmdec],c0_residual=a['pm_icrs_masyr']-[label.pmra,label.pmdec],
                parallax_slope=np.einsum('ani,ni->a',k,prediction-no_pi),negative_parallax_clamped=negative,
                radial_velocity_missing=no_rv,full_gaia_position_covariance=gcov is not None)
            states.append(state)
            split=split_identity(identity);slot=int(entry['source_key'][1]);crowd=float(neighbor[slot])
            for j,index in enumerate(ix):
                det=data.iloc[index];field=str(det.get('field_id','unknown')) if det.survey=='SDSS' else str(det.detector_id)
                physical_group=str(keys[index])+':'+field+':'+str(det.band)
                rawres=xy[index]-prediction[j];c0res=rawres-z[j]
                allmeasure.append(dict(source_key=physical,identity=identity,physical_group=physical_group,calibration_group=str(keys[index]),
                    survey=str(det.survey),instrument=str(det.get('instrument','unknown')),band=str(det.band),time=float(t[index]),
                    color=gi if gi is not None else np.nan,ri=ri,rmag=rmag,crowding_mas=crowd,split=split,
                    ra_deg=float(det.ra_deg),dec_deg=float(det.dec_deg),raw_graph=rawres,c0_graph=c0res,
                    k_icrs=k[:,j,:],
                    airmass=float(det.get('airmass',np.nan)),sigma_graph=cp[index],gaia_graph_covariance=gcov[j] if gcov is not None else None))
        print('STELLAR_POSITION_CORE',row['core'],len(chosen),flush=True)
    measurements=pd.DataFrame(allmeasure);matched=[]
    # One physical detector/field uses one common local basis, so its common C0
    # offset cancels exactly in the red-minus-blue position comparison.
    for name,g in measurements.groupby('physical_group',sort=True):
        center_row=next(r for r in plan['selected'] if r['core']==g.source_key.iat[0][0])
        pivot=forward_jacobian(float(g.ra_deg.median()),float(g.dec_deg.median()),center_row['graph_center']);inv=np.linalg.inv(pivot)
        ind=g.index;raw=np.asarray(g.raw_graph.tolist())@inv.T;c0=np.asarray(g.c0_graph.tolist())@inv.T
        sky=project(g.ra_deg,g.dec_deg,(float(g.ra_deg.median()),float(g.dec_deg.median())))
        measurements.loc[ind,['raw_e','raw_n']]=raw;measurements.loc[ind,['c0_e','c0_n']]=c0
        measurements.loc[ind,['sky_e','sky_n']]=sky
        for j in ind:allmeasure[j]['group_to_graph']=pivot
    pairs,ngroups=colour_pairs(measurements);cancellation=max((np.max(abs(r['raw_difference']-r['c0_difference'])) for r in pairs),default=0.)
    # The position-to-PM test is a diagnostic prediction on held identities;
    # never applied to the science catalogue. Both pair members must be train.
    trainpairs,_=colour_pairs(measurements.loc[measurements.split.lt(3)])
    bygroup=defaultdict(list)
    for r in trainpairs:bygroup[r['physical_group']].append(r['raw_difference']/r['delta_color'])
    coefficients={g:np.median(v,axis=0) for g,v in bygroup.items() if len(v)>=3}
    predictions=defaultdict(lambda:np.zeros(2));support_k=Counter();total_k=Counter()
    for m in allmeasure:
        key=tuple(m['source_key']);amplitude=float(np.linalg.norm(m['k_icrs']));total_k[key]+=amplitude
        if m['physical_group'] in coefficients and np.isfinite(m['color']):
            predictions[key]+=np.asarray(m['k_icrs'])@m['group_to_graph']@coefficients[m['physical_group']]*(m['color']-1.)
            support_k[key]+=amplitude
    held_prediction={}
    for split,name in [(3,'validation'),(4,'confirmation')]:
        heldids={tuple(m['source_key']) for m in allmeasure if m['split']==split}
        use=[r for r in states if tuple(r['source_key']) in heldids and 'c0_residual' in r and support_k[tuple(r['source_key'])]>=.5*total_k[tuple(r['source_key'])]]
        held_prediction[name]=dict(n=len(use),minimum_K_amplitude_support_fraction=.5,
            before=summarize([r['c0_residual'] for r in use]),
            predicted_partial_color_pm=summarize([predictions[tuple(r['source_key'])] for r in use]),
            diagnostic_residual_after_partial_prediction=summarize([np.asarray(r['c0_residual'])-predictions[tuple(r['source_key'])] for r in use]))
    selected=[r for r in states if 'c0_residual' in r];summary={}
    for lo,hi in [(15,19),(19,21)]:
        summary[f'{lo}_{hi}']={}
        for name,select in [('blue',lambda r:r['color'] is not None and r['color']<.5),('red',lambda r:r['color'] is not None and r['color']>=2),('all',lambda r:True)]:
            rows=[r for r in selected if lo<=r['rmag']<hi and select(r)]
            summary[f'{lo}_{hi}'][name]=dict(raw=summarize([r['raw_residual'] for r in rows]),c0=summarize([r['c0_residual'] for r in rows]),
                C0_response=summarize([r['c0_predicted_change'] for r in rows]),annual_parallax_slope=summarize([r['parallax_slope'] for r in rows]))
    pair_summary={}
    for survey in ('SDSS','NSC'):
        rows=[r for r in pairs if r['survey']==survey]
        pair_summary[survey]=dict(raw_red_minus_blue_mas=summarize([r['raw_difference'] for r in rows]),
            c0_red_minus_blue_mas=summarize([r['c0_difference'] for r in rows]),physical_groups=len({r['physical_group'] for r in rows}))
    with gzip.open(output/'private_stellar_positions.json.gz','wt') as f:
        json.dump(dict(states=states,measurements=allmeasure,pairs=pairs),f,default=json_default,allow_nan=True)
    report=dict(frozen_bright_identities=len(bright),bright_processed=len(selected),all_fixed_processed=len(states),
        missing=dict(missing),maximum_C0_K_identity_error_masyr=maximum_k,maximum_common_group_color_cancellation_mas=cancellation,
        bright_motion=summary,matched_within_group_positions=pair_summary,metadata_matched_pairs=len(pairs),physical_color_groups=ngroups,
        training_color_groups_with_three_pairs=len(coefficients),held_identity_partial_K_prediction=held_prediction,
        gaia_queries=queries,full_gaia_covariance_sources=sum(r.get('full_gaia_position_covariance',False) for r in states),
        negative_parallax_sources=sum(r.get('negative_parallax_clamped',False) for r in states),missing_RV_sources=sum(r.get('radial_velocity_missing',False) for r in states),
        elapsed_seconds=monotonic()-start,slurm_job=os.environ['SLURM_JOB_ID'],
        scope='Fixed original bright and optical identities, fixed members; B2 full2D IRLS weights then freeze weights for raw/C0 comparison. Actual TT observation time converted to TCB for Gaia propagation; builtin barycentric Earth AU annual parallax, negative measured parallax clamped to zero and missing RV zero with explicit flags. Published astrometric ICRS directions, no duplicate aberration/precession/refraction. Earth-surface parallax omitted. Full five-parameter Gaia covariance where returned; RV uncertainty not propagated. Within group colour pairs require <=0.05deg separation, <=0.5mag, neighbour>=2arcsec and crowding ratio<=2. No fitted production colour/DCR correction; no zenith-direction metadata, hence colour evidence alone is not a DCR identification.')
    if maximum_k>1.e-3:raise ValueError('C0 slope response identity exceeded registered tolerance')
    check_aggregate(report);save_json(output/'v21_stellar_positions_summary.json',report)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('registration','old_cluster','output','cache'):p.add_argument(n,type=Path)
    a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use Slurm resources')
    run(**vars(a))
