"""Complete the sole archived morphology trial, under its original geometry.

No new selection threshold: recover original training sources, append frozen new
members, then expose the held year to the complete competing prediction set.
"""
import argparse
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import gzip
import json
import os
from pathlib import Path
import re
from time import monotonic

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from astropy.io import fits
from astropy.time import Time
from matplotlib.path import Path as Polygon

from . import access
from .__main__ import tomllib
from .allband_qso import evaluated_fit,physical_candidates
from .astrometry import project,unproject
from .blind_catalogue_quality import read
from .continuous_reference import PRECISION,SPATIAL,read_offsets,reference_frame
from .coordinate_metric import AngularTree
from .dark_precision import query_unique
from .exposure_catalogues import inside
from .independent_year import exclusive
from .io import digest,save_json,json_default
from .nsc_science_catalogue import select_catalogue
from .pilot_production import quality_detections
from .precision_v03 import calibration_groups,calibration_support,fit_with_calibration
from .public_summary import check_aggregate
from .usability_cluster import summarize
from .validation import ratio

CHIP_COLUMNS='exposure,ccdnum,instrument,mjd,filter,airmass,fwhm,nmeas,depth95,depth10sig,'+','.join('vertex_'+axis+str(i) for axis in ('ra','dec') for i in range(1,5))


def metadata_chips(source,cache):
    """Exposure universe comes from independent buffered chip-centre query."""
    initial=pd.read_csv(source/'chips.csv',dtype={'exposure':str});exposures=sorted(initial.exposure.unique())
    parts=[]
    for start in range(0,len(exposures),60):
        where='exposure IN ('+','.join(map(access.literal,exposures[start:start+60]))+')'
        parts.append(cache.checked('nsc',CHIP_COLUMNS,'nsc_dr2.chip',where,cap=30000))
    f=pd.concat(parts,ignore_index=True).drop_duplicates(['exposure','ccdnum']) if parts else pd.DataFrame(columns=CHIP_COLUMNS.split(','))
    return f,dict(independent_chip_center_entries=len(initial),independent_exposures=len(exposures),returned_CCD_entries=len(f),
        input_chip_sha256=digest(source/'chips.csv'),internal_exposure_bad_pixel_masks_available=False,
        scope='All bands, no target-detection existence requirement. Actual polygons and 10sigma catalogue depth. depth95 is a magnitude percentile, not 95percent completeness. Internal effective-pixel masks unavailable; geometry/depth opportunities remain conditional, not verified image extraction completeness.')


def opportunities(data,models,members,chips,center,solution,held_year):
    n=len(models);covered=np.zeros(n,bool);depth=np.zeros(n,bool);five=np.zeros(n,bool);unknown=np.zeros(n,bool)
    counts=np.zeros(n,int);qualified=np.zeros(n,int);bands=[]
    for ix in members:
        f=data.iloc[ix];bands.append({b:float(g.magnitude.median()) for b,g in f.loc[f.survey.eq('NSC')].groupby('band') if g.magnitude.notna().any()})
    positions=np.asarray([m.coeff[0] for m in models]);motion=np.asarray([m.coeff[1] for m in models]);epoch=models[0].epoch if n else 2015.5
    valid=chips.mjd.notna()&chips.mjd.between(50000,62000)
    years=np.zeros(len(chips),int);years[valid]=np.asarray(Time(chips.loc[valid,'mjd'],format='mjd',scale='utc').datetime64,dtype='datetime64[Y]').astype(int)+1970
    invalid=0
    for _,chip in chips.loc[valid&(years==held_year)].iterrows():
        key='NSC:'+str(chip.exposure)
        if key not in solution:continue
        vertices=np.array([[chip[f'vertex_ra{i}'],chip[f'vertex_dec{i}']] for i in range(1,5)],float)
        if not np.isfinite(vertices).all():invalid+=1;continue
        polygon=Polygon(project(*vertices.T,center))
        time=float(Time(chip.mjd,format='mjd',scale='utc').tt.jyear)
        # The fitted positions are C0-corrected; add the archived offset before
        # checking a footprint expressed in the catalogue's published WCS.
        prediction=positions+motion*(time-epoch)+solution[key][0]
        here=polygon.contains_points(prediction);covered|=here;counts+=here
        mag=np.array([r.get(str(chip['filter']),np.nan) for r in bands])
        good=here&np.isfinite(mag)&np.isfinite(chip.depth10sig)
        depth|=good&(mag<=chip.depth10sig);five|=good&(mag<=chip.depth10sig+2.5*np.log10(2))
        unknown|=here&~good;qualified+=good&(mag<=chip.depth10sig)
    return dict(geometric=covered,catalogue_10sigma=depth,background_limited_5sigma_proxy=five,depth_or_band_unknown=unknown,
        geometric_CCD_count=counts,ten_sigma_CCD_count=qualified),dict(invalid_polygon_CCDs=invalid)


def independent_morphology(positions,root,center):
    """Bounded public LS sweeps; shared imaging is explicitly not stellar truth."""
    sky=unproject(positions,center);frames=[];files=0
    for side in ('north','south'):
        folder=root/side/'sweep/9.0'
        if not folder.exists():continue
        for file in sorted(folder.glob('sweep-*.fits')):
            m=re.fullmatch(r'sweep-(\d{3})([mp])(\d{3})-(\d{3})([mp])(\d{3})\.fits',file.name)
            if not m:continue
            # LS sweep names encode integer degrees (p025 is +25deg).
            a,s,d,b,t,e=m.groups();lo=float(d)*(-1 if s=='m' else 1);hi=float(e)*(-1 if t=='m' else 1)
            overlap=(sky[:,0]>=float(a)-.003)&(sky[:,0]<=float(b)+.003)&(sky[:,1]>=lo-.003)&(sky[:,1]<=hi+.003)
            if not overlap.any():continue
            with fits.open(file,memmap=True) as h:
                table=h[1].data
                keep=(table['RA']>=sky[:,0].min()-.003)&(table['RA']<=sky[:,0].max()+.003)&(table['DEC']>=sky[:,1].min()-.003)&(table['DEC']<=sky[:,1].max()+.003)
                columns=('RA','DEC','TYPE','MASKBITS','FRACFLUX_R','FRACMASKED_R','FRACIN_R','NOBS_R')
                frames.append(pd.DataFrame({c.lower():np.asarray(table[c][keep]).astype(str if c=='TYPE' else float) for c in columns if c in table.names}))
                files+=1
    result=[dict(status='no_unique_external_position') for _ in positions]
    if not frames:return result,dict(sweep_files=0,matched=0)
    f=pd.concat(frames,ignore_index=True).drop_duplicates(['ra','dec']);xy=project(f.ra,f.dec,center)
    tree=AngularTree(xy,center);found,near=query_unique(tree,positions,700,300)
    for i in np.flatnonzero(found):
        r=f.iloc[near[i]];result[i]=dict(status='unique_LS_position',shape=str(r.type).strip(),
            masked=bool(r.get('maskbits',0)!=0),blend_proxy=bool(r.get('fracflux_r',0)>.2),
            fractional_mask=r.get('fracmasked_r'),fractional_in=r.get('fracin_r'),r_exposures=r.get('nobs_r'))
    return result,dict(sweep_files=files,matched=int(found.sum()),scope='Independent catalogue processing, partly shared NSC imaging. Published morphology and crowding proxies; not independent images or certified star truth. 0.7arcsec and 0.3arcsec sky uniqueness at static published positions; fast motion can reduce support.')


def repair_morphology(task):
    row,completed,output,lsroot=task;prior=Path(completed)/row['core'];dest=Path(output)/row['core'];dest.mkdir(parents=True,exist_ok=True)
    result=dest/'v21_external_morphology_core_summary.json'
    ticket=dict(competition_sha256=digest(prior/'v21_candidate_completion_core_summary.json'),implementation_sha256=digest(Path(__file__)))
    if result.exists():
        r=read(result)
        if r['ticket']!=ticket:raise ValueError('External morphology audit signature changed')
        return r
    start=monotonic();frozen=read(prior/'private_training_competitors.json.gz');nold=frozen['original'];members=frozen['members'][nold:]
    config=tomllib.loads(PRECISION.read_text());settings=tomllib.loads(SPATIAL.read_text());graph=Path(row['graph'])/'private_c0.npz';center=reference_frame(graph)
    data,xy,t,sigma,keep=quality_detections(Path(row['source']),center,config,settings['production']);keys=calibration_groups(data);lookup=pd.Index(data.detection_id)
    solution=read_offsets(graph);points=[]
    for member in members:
        ix=lookup.get_indexer(member)
        if (ix<0).any():raise ValueError('Frozen training identity missing from raw source')
        model,_=fit_with_calibration(xy[ix],t[ix],sigma[ix],config['reference_epoch'],[solution[g] for g in keys[ix]])
        points.append(model.coeff[0])
    morph,meta=independent_morphology(np.asarray(points).reshape(-1,2),Path(lsroot),center)
    with gzip.open(dest/'private_external_morphology.json.gz','wt') as f:json.dump(morph,f,default=json_default,allow_nan=False)
    r=dict(ticket=ticket,core=row['core'],training_added_sources=len(members),external_morphology_counts=dict(Counter(v.get('shape',v['status']) for v in morph)),
        blend_proxy_count=sum(v.get('blend_proxy',False) for v in morph),mask_flag_count=sum(v.get('masked',False) for v in morph),
        metadata=meta,elapsed_seconds=monotonic()-start,slurm_job=os.environ['SLURM_JOB_ID'],
        scope='Corrected sweep filename degree parser. Fixed old-geometry training members/fit reused; full competition and opportunity results unchanged. Previous external morphology absence was an implementation failure, superseded by this audit. LS shares some NSC images and is not stellar truth.')
    check_aggregate(r);save_json(result,r);print('EXTERNAL_MORPHOLOGY_AUDITED',row['core'],r['external_morphology_counts'],flush=True);return r


def identity_control(task):
    """Metadata-hash original control; joint shape/held-position/depth evidence."""
    from hashlib import sha256
    from .allband_blind_year import r_magnitude
    row,completed,morphology,output,lsroot=task;prior=Path(completed)/row['core'];dest=Path(output)/row['core'];dest.mkdir(parents=True,exist_ok=True)
    reportfile=dest/'v21_candidate_identity_core_summary.json'
    ticket=dict(completion_sha256=digest(prior/'private_candidate_completion.json.gz'),morphology_sha256=digest(Path(morphology)/row['core']/'private_external_morphology.json.gz'),implementation_sha256=digest(Path(__file__)))
    if reportfile.exists():
        previous=read(reportfile)
        if previous['ticket']!=ticket:raise ValueError('Candidate identity signature changed')
        return previous
    frozen=read(prior/'private_candidate_completion.json.gz');nold=frozen['nold'];members=frozen['training_keys'];matched={r['training_slot'] for r in frozen['matches']}
    original=[i for i in range(nold) if int(sha256('|'.join(members[i]).encode()).hexdigest()[:16],16)%10==0]
    config=tomllib.loads(PRECISION.read_text());settings=tomllib.loads(SPATIAL.read_text());graph=Path(row['graph'])/'private_c0.npz';center=reference_frame(graph)
    data,xy,t,sigma,keep=quality_detections(Path(row['source']),center,config,settings['production']);keys=calibration_groups(data);lookup=pd.Index(data.detection_id);nsc=data.survey.eq('NSC').to_numpy();solution=read_offsets(graph)
    points=[]
    for i in original:
        ix=lookup.get_indexer(members[i])
        if (ix<0).any():raise ValueError('Original control member absent')
        m,_=fit_with_calibration(xy[ix],t[ix],sigma[ix],config['reference_epoch'],[solution[g] for g in keys[ix]])
        points.append(m.coeff[0])
    control,_=independent_morphology(np.asarray(points).reshape(-1,2),Path(lsroot),center)
    added=read(Path(morphology)/row['core']/'private_external_morphology.json.gz');shapes={i:m for i,m in zip(original,control)}
    shapes.update({nold+i:m for i,m in enumerate(added)});classes={}
    for name,indices in [('original_hash_control',original),('added',list(range(nold,len(members))))]:
        groups=defaultdict(list)
        for i in indices:
            ix=lookup.get_indexer(members[i]);mag,origin=r_magnitude(ix,data,nsc);m=shapes[i]
            band=str(int(mag)) if np.isfinite(mag) and 20<=mag<23 else 'other_or_missing'
            record=dict(shape=m.get('shape',m['status']),held_unique=i in matched,
                ten_sigma=bool(frozen['opportunities']['catalogue_10sigma'][i]),geometry=bool(frozen['opportunities']['geometric'][i]),
                blend_proxy=m.get('blend_proxy',False),masked=m.get('masked',False))
            groups['all'].append(record);groups[band].append(record)
        classes[name]={}
        for low,values in groups.items():
            psf=[r for r in values if r['shape']=='PSF'];supported=[r for r in psf if r['ten_sigma']]
            classes[name][low]=dict(n=len(values),shape_counts=dict(Counter(r['shape'] for r in values)),
                unique_held_sources=sum(r['held_unique'] for r in values),PSF_with_unique_held=sum(r['held_unique'] for r in psf),
                PSF_with_ten_sigma=len(supported),PSF_ten_sigma_unique_held=sum(r['held_unique'] for r in supported),
                blend_proxy_count=sum(r['blend_proxy'] for r in values),mask_flag_count=sum(r['masked'] for r in values))
    r=dict(core=row['core'],ticket=ticket,classes=classes,slurm_job=os.environ['SLURM_JOB_ID'],
        scope='Original training control sampled by physical member SHA modulo10 before external morphology, not PM. Added sample complete. NSC r preferred, otherwise SDSS r for magnitude; fixed training identities and joint held matching. LS PSF/extended classifications are independent-processing proxies with shared imaging, not a stellar-purity gold sample; no new production threshold.')
    check_aggregate(r);save_json(reportfile,r);print('CANDIDATE_IDENTITY_CONTROL',row['core'],len(original),len(added),flush=True);return r


def one(task):
    row,archive,output,cache,lsroot=task;archive=Path(archive)/row['core'];dest=Path(output)/row['core'];dest.mkdir(parents=True,exist_ok=True)
    reportfile=dest/'v21_candidate_completion_core_summary.json';source=Path(row['source'])
    ticket=dict(source_sha256=digest(source/'run_manifest.json'),old_candidate_sha256=digest(archive/'v20_selection_candidate_core_summary.json'),implementation_sha256=digest(Path(__file__)))
    if reportfile.exists():
        old=read(reportfile)
        if old['ticket']!=ticket:raise ValueError('Candidate completion signature changed')
        return old
    start=monotonic();config=tomllib.loads(PRECISION.read_text());settings=tomllib.loads(SPATIAL.read_text())
    spec=tomllib.loads(Path(row['geometry']).read_text())['fields'][row['core']];graph=Path(row['graph'])/'private_c0.npz';center=reference_frame(graph)
    solution,gauges=read_offsets(graph,with_gauge=True);solution={k:v for k,v in solution.items() if gauges[k]==row['reference_gauge']}
    data,xy,t,sigma,keep=quality_detections(source,center,config,settings['production']);keys=calibration_groups(data)
    nsc=data.survey.eq('NSC').to_numpy();support=calibration_support(keys,solution);physical=physical_candidates(data)
    prior=read(archive/'v20_selection_candidate_core_summary.json');held_year=prior['samples']['training']['held_year']['year']
    years=np.asarray(Time(t,format='jyear',scale='tt').utc.datetime64,dtype='datetime64[Y]').astype(int)+1970;held=nsc&(years==held_year)
    retain=~held;d=data.loc[retain].reset_index(drop=True);x=xy[retain];time=t[retain];err=sigma[retain];k=keys[retain];rkeep=keep[retain]
    @lru_cache(maxsize=32768)
    def solve(ix):
        index=np.asarray(ix,int);return fit_with_calibration(x[index],time[index],err[index],config['reference_epoch'],[solution[g] for g in k[index]])
    def fit(ix):return solve(tuple(ix))
    def final(ix):return evaluated_fit(ix,d,x,time,err,k,solution,center,config['reference_epoch'],model=fit(ix)[0])
    _,original,_,_=select_catalogue(d,x,time,err,rkeep,solution,k,center,spec,config,settings,fit,final)
    original={tuple(sorted(d.detection_id.iloc[r['index']])):r for r in original}
    if len(original)!=prior['samples']['training']['original']:raise ValueError('Original training universe did not reproduce')
    added=read(archive/'private_training_candidate_members.json.gz');added_keys=[tuple(sorted(r['detection_ids'])) for r in added]
    if set(added_keys)&set(original):raise ValueError('Archived new and old training identities overlap')
    allkeys=list(sorted(original))+sorted(added_keys);nold=len(original);lookup=pd.Index(data.detection_id);local=pd.Index(d.detection_id)
    models=[];members=[];magnitudes=[]
    for member in allkeys:
        ix=local.get_indexer(member)
        if (ix<0).any():raise ValueError('Held year or absent member in frozen training set')
        models.append(fit(tuple(ix))[0]);members.append(lookup.get_indexer(member))
        magnitudes.append(float(d.loc[ix[d.band.iloc[ix].eq('r').to_numpy()],'magnitude'].median()) if np.any(d.band.iloc[ix].eq('r')) else None)
    # Written before any held target match is opened. Group identities fixed.
    with gzip.open(dest/'private_training_competitors.json.gz','wt') as f:json.dump(dict(original=nold,members=allkeys),f,default=json_default,allow_nan=False)
    positions=np.asarray([m.coeff[0] for m in models]);motions=np.asarray([m.coeff[1] for m in models]);epoch=config['reference_epoch']
    reproduced=set();oldonly=set();newonly=set();null=[set() for _ in range(4)];collisions=0;residual=[];matches=[]
    for exposure,frame in data.loc[held&physical&support].groupby('exposure_id',sort=False):
        ix=frame.index.to_numpy();mid=float(np.median(t[ix]));coord=xy[ix]-np.asarray([solution[g][0] for g in keys[ix]])
        pred=positions+motions*(mid-epoch);tree=cKDTree(coord)
        good,nearest,c=exclusive(tree,pred,1000,settings['production']['competitor_margin_mas']);collisions+=c
        reproduced.update(np.flatnonzero(good));a,_,_=exclusive(tree,pred[:nold],1000,settings['production']['competitor_margin_mas']);oldonly.update(np.flatnonzero(a))
        a,_,_=exclusive(tree,pred[nold:],1000,settings['production']['competitor_margin_mas']);newonly.update(nold+np.flatnonzero(a))
        for j in np.flatnonzero(good):
            difference=coord[nearest[j]]-models[j].predict(t[ix[nearest[j]]]);residual.append(difference)
            matches.append(dict(training_slot=int(j),held_detection=str(data.detection_id.iat[ix[nearest[j]]]),position_residual=difference))
        for shift,store in zip([(30000,10000),(-30000,10000),(10000,-30000),(-10000,-30000)],null):
            good,_,_=exclusive(tree,pred+shift,1000,settings['production']['competitor_margin_mas']);store.update(np.flatnonzero(good))
    chips,chipmeta=metadata_chips(source,access.Cache(Path(cache)));chips.to_csv(dest/'private_opportunity_chips.csv.gz',index=False)
    opportunity,opmeta=opportunities(data,models,members,chips,center,solution,held_year)
    morphology,morphmeta=independent_morphology(positions[nold:],Path(lsroot),center)
    # Full-epoch catalogue identity audit is separate from held-year prediction.
    # Original and new sources also compete jointly for each spectral label.
    baseline=read(Path(row['fit'])/'private_standard_motion.json.gz')
    fullnew=read(archive/'private_full_candidate_members.json.gz')
    original_points=project(*np.asarray([r['position_deg'] for r in baseline]).T,center)
    fullpoints=list(original_points);fullmotion=[r['pm_icrs_masyr'] for r in baseline]
    for r in fullnew:
        ix=lookup.get_indexer(r['detection_ids'])
        if (ix<0).any():raise ValueError('Full candidate member absent')
        model,_=fit_with_calibration(xy[ix],t[ix],sigma[ix],config['reference_epoch'],[solution[g] for g in keys[ix]])
        fullpoints.append(model.coeff[0]);fullmotion.append(evaluated_fit(ix,data,xy,t,sigma,keys,solution,center,config['reference_epoch'],model=model)['pm_icrs_masyr'])
    spectra=pd.read_csv(source/'spectra_union.csv');spectra=spectra.loc[inside(spectra.ra,spectra.dec,spec)].reset_index(drop=True)
    tree=cKDTree(project(spectra.ra,spectra.dec,center));identity={}
    for name,pts in [('original',original_points),('candidate',np.asarray(fullpoints))]:
        good,near=query_unique(tree,pts,500,300);identity[name]={int(near[j]):int(j) for j in np.flatnonzero(good)}
    qso={}
    for low in (21,22):
        orig={k for k in identity['original'] if low<=spectra.reference_r.iat[k]<low+1}
        current={k for k in identity['candidate'] if low<=spectra.reference_r.iat[k]<low+1}
        new={k for k in current if identity['candidate'][k]>=len(baseline)}
        qso[str(low)]=dict(original_labels=len(orig),candidate_labels=len(current),lost_original_labels=len(orig-current),
            added_labels=len(new),added_motion=summarize([fullmotion[identity['candidate'][k]] for k in sorted(new)]),
            common_original=summarize([baseline[identity['original'][k]]['pm_icrs_masyr'] for k in sorted(orig&current)]),
            common_candidate=summarize([fullmotion[identity['candidate'][k]] for k in sorted(orig&current)]))
    strata={}
    for name,ix in [('original',np.arange(nold)),('added',np.arange(nold,len(models)))]:
        by_op={}
        for tag in ('geometric','catalogue_10sigma','background_limited_5sigma_proxy','depth_or_band_unknown'):
            choices=set(ix[opportunity[tag][ix]]);by_op[tag]=dict(candidates=len(choices),unique_reproductions=len(choices&reproduced),ratio=ratio(len(choices&reproduced),len(choices)))
        strata[name]=dict(training_sources=len(ix),unique_reproductions=len(set(ix)&reproduced),shifted_unique_sources=[len(set(ix)&s) for s in null],opportunities=by_op)
    morphcounts=Counter(r.get('shape',r['status']) for r in morphology);blends=sum(r.get('blend_proxy',False) for r in morphology)
    with gzip.open(dest/'private_candidate_completion.json.gz','wt') as f:json.dump(dict(training_keys=allkeys,nold=nold,matches=matches,
        opportunities=opportunity,external_morphology=morphology,training_magnitude=magnitudes),f,default=json_default,allow_nan=False)
    report=dict(ticket=ticket,core=row['core'],training=strata,old_training_count_reproduced=True,held_year=held_year,
        original_only_unique=len(oldonly),original_lost_after_full_competition=len(oldonly-reproduced),
        added_only_unique=len(newonly),added_lost_after_full_competition=len(newonly-reproduced),competing_claims_rejected=collisions,
        matched_position_residual_mas=summarize(residual),chip_metadata=dict(**chipmeta,**opmeta),external_morphology_counts=dict(morphcounts),
        external_blend_proxy_count=blends,external_morphology=morphmeta,elapsed_seconds=monotonic()-start,slurm_job=os.environ['SLURM_JOB_ID'],
        full_epoch_spectral_competition=qso,
        scope='Original geometry/C0/W0 and sole0.6 threshold; all original+frozen new training sources compete. Test-year excluded before original association. Geometry and catalogue depth denominators measured; internal CCD effective-pixel masks absent and stellar purity truth remains limited. Published QSO recovery is stationary-source evidence, not star certification.')
    check_aggregate(report);save_json(reportfile,report);print('CANDIDATE_COMPLETION_CORE',row['core'],nold,len(added),len(reproduced),flush=True);return report


def run(registration,archive,output,cache,lsroot,workers):
    plan=read(registration);output.mkdir(parents=True,exist_ok=True)
    tasks=[(r,str(archive),str(output),str(cache/r['core']),str(lsroot)) for r in plan['selected']]
    with ProcessPoolExecutor(max_workers=workers) as pool:cores=list(pool.map(one,tasks))
    report=dict(cores=cores,slurm_job=os.environ['SLURM_JOB_ID'],
        decision_state='assessed_with_remaining_external_support_limits',
        scope='Full source competition and measured CCD/depth opportunity audits complete. Shared-image external morphology is evidence, not independent stellar identity truth. Adoptable scope requires final stratified review of actual results; no second threshold searched.')
    check_aggregate(report);save_json(output/'v21_candidate_completion_summary.json',report)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('registration','archive','output','cache','lsroot'):p.add_argument(n,type=Path)
    p.add_argument('--workers',type=int,default=2);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use Slurm resources')
    run(**vars(a))
