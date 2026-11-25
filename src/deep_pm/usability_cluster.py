"""NGC 5466: frozen old targets, independently selected deep targets, and stars."""
import argparse
from collections import Counter,defaultdict
import gzip
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .astrometry import project
from .blind_catalogue_quality import read
from .cluster_motion import describe,mean,support_predictions
from .io import digest,save_json,json_default
from .ngc5466_demo import CENTER,TARGET,BACKGROUND,footprints,optical,register,pair_photometry,match_xiao
from .nsc_ownership_overlay import owned_records
from .public_summary import check_aggregate
from .usability_quality import split_identity
from .validation_audit import gaia_bright
from .exposure_catalogues import inside
from .dark_precision import query_unique

REFERENCE=np.array([-5.336,-.809])


def summarize(motion,error=None,truth=None):
    out=dict(n=len(motion))
    if len(motion)<3:return out
    data=np.asarray(motion,float)
    if truth is not None:data=data-np.asarray(truth)
    center=np.median(data,axis=0)
    out.update(median=center,nmad=1.4826*np.median(abs(data-center),axis=0),
        absolute_radial_p68=np.quantile(np.linalg.norm(data,axis=1),.68),equal_mean=np.mean(data,axis=0))
    rng=np.random.default_rng(546620)
    bootstrap=[np.median(data[rng.integers(0,len(data),len(data))],axis=0) for _ in range(256)]
    out['median_object_interval95']=np.quantile(bootstrap,[.025,.975],axis=0).T
    if error is not None:
        error=np.asarray(error,float)
        out.update(within_1sigma=np.mean(abs(data)<=error,axis=0),within_2sigma=np.mean(abs(data)<=2*error,axis=0))
    return out


def uniquely_paired(phot,records):
    matches=pair_photometry(phot,records)
    count=Counter((r[0],r[1]) for r in matches.values())
    return {i:r for i,r in matches.items() if count[(r[0],r[1])]==1},sum(v>1 for v in count.values())


def run(frozen,sweep,photometry,old_summary,xiao,output,xiao_catalogue=None):
    output.mkdir(parents=True,exist_ok=True)
    old=optical(sweep);old.attrs['photometry_sha256']=digest(sweep)
    old,_=register(old,output/'old_photometry')
    # The independent deep template and masks were saved before opening PMs.
    registration=read(photometry/'sample_registration.json')
    if registration['photometry_sha256']!=digest(sweep):raise ValueError('Deep and old optical inputs differ')
    deep=pd.read_csv(photometry/'private_photometric_parent.csv.gz')
    records=[];paths={};specs={};qso={};gaia_frames=[]
    for owned,boxes,cores in footprints(frozen):
        index=read(owned/'fits.json')
        for core in cores:
            for origin,slot,row in owned_records(owned,core):
                if np.linalg.norm(project(*row['position_deg'],CENTER)[0])>1.15*3600000:continue
                records.append((origin,slot,row));paths[origin]=index[origin];specs[origin]=boxes[origin]
            folder=Path(index[core]);source=(folder/'source_receipt.json').resolve().parent
            spectra=pd.read_csv(source/'spectra_union.csv')
            spectra=spectra.loc[inside(spectra.ra,spectra.dec,boxes[core])].reset_index(drop=True)
            for row in read(folder/'private_blind_qso.json'):
                if row['magnitude'] is None:continue
                target=spectra.iloc[row['qso_local_index']]
                qso[(core,row['source_slot'])]=dict(identity=(str(target.origin),float(target.ra),float(target.dec)),spectral_r=row['magnitude'])
            frame=pd.read_csv(source/'validation/gaia.csv',dtype={'source_id':str})
            gaia_frames.append(frame)
    by_key={(origin,slot):row for origin,slot,row in records}
    if len(by_key)!=len(records):raise ValueError('NGC source identity repeated after final ownership')
    qso={k:r for k,r in qso.items() if k in by_key}
    old_match=pair_photometry(old,records)
    deep_match,duplicate_claims=uniquely_paired(deep,records)
    selected={}
    for tag,phot,matched,flag in [('old',old,old_match,'cmd'),('deep',deep,deep_match,'cmd_v20')]:
        for area,(lo,hi) in [('target',TARGET),('background',BACKGROUND)]:
            for lower in np.arange(18,23,.5):
                mask=phot.radius.between(lo,hi,inclusive='left')&phot.r0.between(lower,lower+.5,inclusive='left')&phot[flag]
                selected[f'{tag}__{area}__{lower}']=[(matched[i][0],matched[i][1]) for i in np.flatnonzero(mask) if i in matched]
    old_faint=sum((selected[f'old__target__{lower}'] for lower in np.arange(21,23,.5)),[])
    prior=read(old_summary)['cohorts']['target']['faint']
    if len(old_faint)!=prior['matched_frozen_motion']:raise ValueError('Old NGC photometric cohort changed')
    cat=pd.concat(gaia_frames,ignore_index=True).drop_duplicates('source_id')
    position_ok=cat[['ra','dec']].notna().all(axis=1)
    cat=cat.loc[position_ok&gaia_bright(cat,np.zeros(len(cat)))&cat.ruwe.lt(1.4)].reset_index(drop=True)
    r_anchors=[(k,row) for k,row in by_key.items() if row['magnitude_source']=='NSC_r' and row['magnitude'] is not None and row['magnitude']<21]
    bright={}
    if len(cat) and r_anchors:
        coords=np.asarray([row['position_deg'] for _,row in r_anchors])
        good,near=query_unique(cKDTree(project(*coords.T,CENTER)),project(cat.ra,cat.dec,CENTER),1000,300)
        claims=Counter(near[good])
        for i in np.flatnonzero(good):
            key=r_anchors[near[i]][0]
            if claims[near[i]]==1 and key not in qso:
                bright[key]={field:cat.iloc[i][field] for field in ('source_id','pmra','pmdec','pmra_error','pmdec_error','phot_g_mean_mag','ruwe','visibility_periods_used') if pd.notna(cat.iloc[i][field])}
    wanted=set(qso)|set(bright)|{k for v in selected.values() for k in v}
    details=describe([(a,b,by_key[(a,b)]) for a,b in sorted(wanted)],paths,specs)
    mean_old=mean([details[k] for k in old_faint],reference_motion=REFERENCE)
    np.testing.assert_allclose(mean_old['weighted_mean'],prior['candidate_motion']['weighted_icrs_mean'],atol=1.e-9,rtol=1.e-9)
    paired_xiao,xiao_hash=match_xiao(records,xiao,xiao_catalogue)
    qkeys=sorted(qso);allkeys=sorted(details);lookup={k:i for i,k in enumerate(allkeys)}
    train=[lookup[k] for k in qkeys if split_identity(qso[k]['identity'])<3]
    predictions=support_predictions([details[k] for k in allkeys],train)
    old_zero=np.median([by_key[k]['pm_icrs_masyr'] for k in qkeys],axis=0)
    validation={}
    for residue,tag in [(3,'identity_validation'),(4,'identity_confirmation')]:
        keys=[k for k in qkeys if split_identity(qso[k]['identity'])==residue and lookup[k] in predictions]
        before=[by_key[k]['pm_icrs_masyr'] for k in keys]
        after=[np.asarray(by_key[k]['pm_icrs_masyr'])-predictions[lookup[k]]['predicted_zero'] for k in keys]
        before,after=summarize(before),summarize(after)
        decision=False
        if len(keys)>=20:
            a=np.abs(before['median']);b=np.abs(after['median']);offending=a>.5
            decision=bool(np.all(np.where(offending,b<=.7*a,b<=a+.5)) and after['absolute_radial_p68']<=1.05*before['absolute_radial_p68'])
        validation[tag]=dict(n=len(keys),before=before,after=after,zero_and_p68_gate=decision)
    cohort={};candidate_bias=[]
    for name,keys in selected.items():
        values=[details[k] for k in keys]
        stats=dict(motion=mean(values,reference_motion=REFERENCE))
        pairs=[(by_key[k],paired_xiao.get(id(by_key[k]))) for k in keys]
        pairs=[(row,value) for row,value in pairs if value is not None]
        stats['Xiao_pair']=dict(n=len(pairs))
        if len(pairs)>=3:
            a=np.asarray([r['pm_icrs_masyr'] for r,_ in pairs]);b=np.asarray([v[:2] for _,v in pairs])
            stats['Xiao_pair'].update(c0_equal_mean=a.mean(axis=0),xiao_equal_mean=b.mean(axis=0),difference=summarize(a-b))
        stats['variants']={}
        for variant in ('physical_r_members','nsc_only_members'):
            paired=[r for r in values if variant in r['variants']]
            a=[r['row']['pm_icrs_masyr'] for r in paired];b=[r['variants'][variant]['motion'] for r in paired]
            stats['variants'][variant]=dict(n=len(paired),allband=summarize(a),variant=summarize(b),
                variant_minus_all=summarize(np.asarray(b)-a) if paired else dict(n=0))
        matched=[k for k in keys if lookup[k] in predictions]
        stats['response_matched_qso']=dict(n=len(matched),outside_support=len(keys)-len(matched))
        if len(matched)>=3:
            raw=mean([details[k] for k in matched],reference_motion=REFERENCE)
            weight=1/np.mean([np.asarray(by_key[k]['pm_error_masyr'])**2 for k in matched],axis=1);weight/=weight.sum()
            shift=weight@np.asarray([predictions[lookup[k]]['predicted_zero'] for k in matched])
            stats['response_matched_qso'].update(raw=raw,training_qso_predicted_shift=shift,
                diagnostic_translated_mean=raw['weighted_mean']-shift,
                diagnostic_translated_offset=raw['weighted_mean']-shift-REFERENCE)
        galaxy=[r['galaxy_predicted_pm'] for r in values if r['galaxy_predicted_pm'] is not None]
        stats['held_galaxy_predicted_motion']=summarize(galaxy)
        stats['known_spectral_qso_in_candidates']=sum(k in qso for k in keys)
        cohort[name]=stats
    bright_rows=[details[k] for k in bright]
    bright_residual=np.array([by_key[k]['pm_icrs_masyr']-np.array([bright[k]['pmra'],bright[k]['pmdec']]) for k in bright]).reshape(-1,2)
    bright_errors=np.array([np.hypot(by_key[k]['pm_error_masyr'],[bright[k]['pmra_error'],bright[k]['pmdec_error']]) for k in bright]).reshape(-1,2)
    stellar=dict(all=summarize(bright_residual,bright_errors),color_bins={})
    for lower,upper in [(-1,0),(0,.5),(.5,1),(1,1.5),(1.5,2),(2,4)]:
        use=[j for j,r in enumerate(bright_rows) if r['sdss_g_minus_i'] is not None and lower<=r['sdss_g_minus_i']<upper]
        stellar['color_bins'][str(lower)]=dict(lower_gi=lower,upper_gi=upper,residual=summarize(bright_residual[use],bright_errors[use]))
    old_bright=sum((selected[f'old__target__{lower}'] for lower in np.arange(18,21,.5)),[])
    bright_mean=mean([details[k] for k in old_bright],reference_motion=REFERENCE)
    all_mean=mean([details[k] for k in old_bright+old_faint],reference_motion=REFERENCE)
    variance_ratio=np.diag(all_mean['independent_covariance'])/np.diag(bright_mean['independent_covariance'])
    bright_matched=[k for k in bright if lookup[k] in predictions]
    bright_before=[np.asarray(by_key[k]['pm_icrs_masyr'])-[bright[k]['pmra'],bright[k]['pmdec']] for k in bright_matched]
    bright_after=[r-predictions[lookup[k]]['predicted_zero'] for k,r in zip(bright_matched,bright_before)]
    bright_test=dict(n=len(bright_matched),before=summarize(bright_before),after=summarize(bright_after))
    bright_gate=False
    if len(bright_matched)>=20:
        a=np.abs(bright_test['before']['median']);b=np.abs(bright_test['after']['median'])
        bright_gate=bool(np.all(np.where(a>.5,b<=.7*a,b<=a+.5)) and bright_test['after']['absolute_radial_p68']<=1.05*bright_test['before']['absolute_radial_p68'])
    old_faint_matched=[k for k in old_faint if lookup[k] in predictions]
    cluster_test=dict(n=len(old_faint_matched));cluster_gate=False
    if len(old_faint_matched)>=20:
        raw=mean([details[k] for k in old_faint_matched],reference_motion=REFERENCE)
        w=1/np.mean([np.asarray(by_key[k]['pm_error_masyr'])**2 for k in old_faint_matched],axis=1);w/=w.sum()
        shift=w@np.asarray([predictions[lookup[k]]['predicted_zero'] for k in old_faint_matched])
        a=np.abs(raw['offset_from_bright_reference']);translated=raw['weighted_mean']-shift-REFERENCE;b=np.abs(translated)
        cluster_gate=bool(np.all(np.where(a>.5,b<=.7*a,b<=a+.5)))
        cluster_test.update(raw=raw,training_predicted_shift=shift,translated_offset=translated)
    adopted=bool(all(v['zero_and_p68_gate'] for v in validation.values()) and bright_gate and cluster_gate)
    # Tuple-key mapping is encoded as ordered rows in the private diagnostic.
    with gzip.open(output/'private_cluster_diagnostics.json.gz','wt') as f:
        json.dump(dict(entries=[dict(source_key=k,**details[k],qso=qso.get(k),gaia=bright.get(k)) for k in allkeys],
            selected=selected,predictions=predictions),f,default=json_default,allow_nan=False)
    phot_summary=read(photometry/'v20_ngc_photometry_summary.json')
    areas=registration['effective_area_sqdeg'];alpha=areas['target']/areas['evaluation_background']
    mixtures={}
    for low in np.arange(21.,23.,.5):
        target=selected[f'deep__target__{low}'];background=selected[f'deep__background__{low}']
        t=phot_summary['cohorts']['target'][str(low)];b=phot_summary['cohorts']['evaluation_background'][str(low)]
        excess=len(target)-alpha*len(background)
        result_mix=dict(area_ratio=alpha,photometric_candidate_excess=t['cmd_candidates']-alpha*b['cmd_candidates'],
            motion_available_excess=excess,target_motion_recovery=len(target)/t['cmd_candidates'] if t['cmd_candidates'] else None,
            background_motion_recovery=len(background)/b['cmd_candidates'] if b['cmd_candidates'] else None,
            background_mixture_motion=mean([details[k] for k in target+background],np.r_[np.ones(len(target)),np.full(len(background),-alpha)],REFERENCE) if excess>=5 else dict(n=len(target)+len(background)),
            scope='Conditional observed CMD/motion excess, not certified members. Different motion recovery and sampling in target/background remain explicit.')
        mixtures[str(low)]=result_mix
    result=dict(original_faint_cohort_reproduced=True,original_faint=mean_old,original_faint_qso_subtraction=mean_old['weighted_mean']-old_zero,
        local_qso=dict(n=len(qso),raw=summarize([by_key[k]['pm_icrs_masyr'] for k in qkeys]),training=len(train),validation=validation),
        local_candidate_adopted=adopted,local_candidate_stellar_bright_gate=bright_gate,local_candidate_cluster_gate=cluster_gate,local_candidate_bright_validation=bright_test,local_candidate_cluster_validation=cluster_test,
        old_bright_mean=bright_mean,old_combined_mean=all_mean,independent_variance_reduction=1-variance_ratio,independent_standard_deviation_reduction=1-np.sqrt(variance_ratio),
        cohorts=cohort,bright_gaia=stellar,deep_photometry=phot_summary,deep_background_mixture=mixtures,
        rejected_duplicate_photometric_claims=duplicate_claims,
        bright_reference_masyr=REFERENCE,bright_reference_distance_kpc=16.12,
        xiao_index_sha256=xiao_hash,slurm_job=os.environ['SLURM_JOB_ID'],
        scope='Fixed old photometric cohort and separately frozen spatially independent deep template. Actual physical-group K response and synchronous 48 reference draws; Gaia used only in strict bright position-identified validation. QSO translation is diagnostic unless both held identity gates pass; stellar transfer remains separately tested. Candidate cluster means include foreground/background contamination; not certified membership. Xiao shares SDSS inputs and its difference is consistency, not independent truth.')
    check_aggregate(result);save_json(output/'v20_ngc_motion_summary.json',result)
    print('NGC_MOTION_DONE',len(old_faint),len(bright),len(qso),validation,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('frozen','sweep','photometry','old_summary','xiao','output'):p.add_argument(n,type=Path)
    p.add_argument('--xiao-catalogue',type=Path);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use Slurm compute nodes')
    run(**vars(a))
