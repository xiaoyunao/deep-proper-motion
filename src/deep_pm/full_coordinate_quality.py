"""Fixed old identities and final new blind-QSO quality are separate samples."""
import argparse
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path

import healpy as hp
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .astrometry import basis
from .blind_catalogue_quality import read
from .exposure_catalogues import inside,coordinates
from .full_coordinate_sources import write_rows
from .io import digest,save_json
from .nsc_comparison import comparison
from .continuous_summary import estimate
from .public_summary import check_aggregate
from .usability_accounting import index as old_index
from .validation import ratio
from .stream_json import array_items

SETTINGS=dict(min_publish_count=5,tail_masyr=5,bootstrap_draws=800,bootstrap_absolute_zero=True,seed=2226)


def information(row):
    info=row['information'];sky=row['position_deg'];center=row.get('projection_center_deg')
    angle=float(np.degrees(np.arccos(np.clip(basis(*sky)[0]@basis(*center)[0],-1,1)))) if center is not None else None
    return dict(sdss_runs=info['sdss_physical_runs'],nsc_nights=info['nsc_nights'],nsc_seasons=info['nsc_seasons'],
        joint_span=info['joint_span_years'],nsc_span=info['nsc_span_years'],
        leverage=max(info['largest_information_fraction']['physical_group']),bands='+'.join(row['bands']),
        instruments='+'.join(row['instruments']),kind=row['kind'],chart_angle_deg=angle,
        pixel64=int(hp.ang2pix(64,*sky,lonlat=True)),exposures=row['exposures'],nights=row['nights_utc18'] if 'nights_utc18' in row else [],
        sdss_years=info.get('sdss_unique_UTC_years'),nsc_years=info.get('nsc_unique_UTC_years'),
        nsc_epoch_days=sorted({str(n) for e,n in zip(row['exposures'],row.get('nights_utc18',[])) if e.startswith('NSC:')}))


def source_pool(task):
    i,entry,root,old_entries,quarantined=task;root=Path(root);core=entry['core'];out=root/'quality/cores'/core;out.mkdir(parents=True,exist_ok=True)
    source=Path(entry['source']);folder=root/'sources'/core;before=old_entries[core]
    ticket=dict(source_summary_sha256=digest(folder/'v22_source_core_summary.json'),index_sha256=digest(root/'catalogue/index'/core/'private_index_receipt.json'),
        implementation_sha256=digest(Path(__file__)),input_reader_sha256=digest(Path(__file__).with_name('stream_json.py')))
    marker=out/'private_pool_receipt.json'
    if marker.exists():
        old=read(marker)
        if old['ticket']!=ticket or old['sha256']!=digest(out/'private_qso_pool.json.gz'):raise ValueError('Changed QSO readout checkpoint')
        return old
    fixed_entries=read(folder/'private_fixed_blind_qso.json.gz');blind_entries=read(folder/'private_blind_qso.json')
    old_wanted={r['source_slot'] for r in fixed_entries};new_wanted={r['source_slot'] for r in blind_entries}
    oldrows={j:r for j,r in enumerate(array_items(Path(before['path'])/'private_standard_motion.json.gz')) if j in old_wanted}
    newrows={j:r for j,r in enumerate(array_items(folder/'private_standard_motion.json.gz')) if j in new_wanted}
    spectra=pd.read_csv(source/'spectra_union.csv');spectra=spectra.loc[inside(spectra.ra,spectra.dec,entry['spec'])].reset_index(drop=True)
    fixed_groups=defaultdict(list)
    for r in fixed_entries:fixed_groups[r['source_slot']].append(r)
    fixed=[];blind=[];exits=Counter()
    for slot,options in fixed_groups.items():
        # The old scientific universe uses its frozen owner overlay, independent
        # of a new quality gate or whether any new source was discovered.
        if before['moves'].get(slot,core) is None:exits['old_exterior']+=1;continue
        computed=[r for r in options if r['status']=='computed']
        if len(computed)>1:raise ValueError('One old identity computed under incompatible reference components')
        r=dict(computed[0] if computed else options[0]);mag=r['magnitude']
        old=oldrows[slot];qso=spectra.iloc[r['qso_local_index']]
        if mag is not None:assert float(qso.reference_r)==mag
        r.update(identity=(str(qso.origin),float(qso.ra),float(qso.dec)),window=entry['window'],**information(dict(old,projection_center_deg=entry['center'])))
        r['information_epoch_definition']='Original fixed physical members and archived source conditions; new local support/formal exits retained.'
        if r['status']=='computed':r['pm_covariance']=np.asarray(r['new_fit']['covariance'])[np.ix_([1,3],[1,3])]
        fixed.append(r)
    compact=pd.read_parquet(root/'catalogue/index'/core/'science.parquet',columns=['slot','owner'])
    retained=set(compact.loc[compact.owner.notna(),'slot'])
    for r in blind_entries:
        slot=r['source_slot'];src=(i<<32)+slot
        if slot not in retained:exits['new_exterior_spectral']+=1;continue
        if src in quarantined:exits['new_global_identity_excluded']+=1;continue
        row=newrows[slot];s=dict(r);s.update(window=entry['window'],**information(row))
        s['pm_covariance']=np.asarray(row['covariance'])[np.ix_([1,3],[1,3])]
        blind.append(s)
    write_rows(out/'private_qso_pool.json.gz',dict(fixed=fixed,blind=blind))
    result=dict(ticket=ticket,core=core,fixed=len(fixed),blind=len(blind),exits=dict(exits),sha256=digest(out/'private_qso_pool.json.gz'))
    save_json(marker,result);print('QSO_CORE_POOL',core,len(fixed),len(blind),flush=True);return result


def unique_positions(rows,tolerance_mas=.036):
    """Reconcile CSV float round trips; no scientific matching-radius change."""
    bad=set();byorigin=defaultdict(list)
    for i,r in enumerate(rows):byorigin[r['identity'][0]].append(i)
    for indices in byorigin.values():
        points=np.array([rows[i]['identity'][1:] for i in indices]);tree=cKDTree(coordinates(*points.T))
        chord=2*np.sin(tolerance_mas/3600000*np.pi/180/2)
        for a,b in tree.query_pairs(chord):bad.update((indices[a],indices[b]))
    return [r for i,r in enumerate(rows) if i not in bad],len(bad)


def reference_common(rows):
    selected=[r for r in rows if 'synchronous_reference_icrs_state_draws' in r or ('synchronous_reference_state_draws' in r and 'new_fit' in r)]
    if len(selected)<5:return dict(n=len(selected))
    grouped=defaultdict(list)
    for r in selected:grouped[r['reference_model_sha256']].append(r)
    n=len(selected);mean_cov=np.zeros((2,2));pieces=[]
    for checksum,part in grouped.items():
        draw=np.array([r['synchronous_reference_icrs_state_draws'] if 'synchronous_reference_icrs_state_draws' in r else np.asarray(r['synchronous_reference_state_draws'])@np.asarray(r['new_fit']['state_jacobian']).T for r in part])[:,:,[1,3]]
        draw-=draw.mean(axis=1,keepdims=True);means=draw.mean(axis=0)
        mean_cov+=(len(part)/n)**2*np.cov(means,rowvar=False)
        points=np.array([r['new_values'][0] if 'new_values' in r else r['values']['new_all'][0] for r in part])
        medians=np.array([np.median(points+draw[:,j],axis=0) for j in range(48)])
        pieces.append(dict(n=len(part),reference_sha256=checksum,mean_common_SE=np.std(means,axis=0,ddof=1),
            median_common_sensitivity_SE=np.std(medians,axis=0,ddof=1)))
    return dict(n=n,independent_reference_components=len(grouped),equal_mean_common_SE=np.sqrt(np.diag(mean_cov)),components=pieces,
        scope='48 common draw labels aligned WITHIN a reference solution. Mean covariance summed across independent reference components with squared number weights, no source-count sqrt. Component median sensitivity recomputes median after each synchronous perturbation. Different-component draw labels are not treated as correlated.')


def ellipse_coverage(points,covariances):
    points=np.asarray(points);covariances=np.asarray(covariances);precision=np.linalg.inv(covariances)
    center=np.median(points,axis=0);out={}
    for name,residual in [('raw',points),('centered_diagnostic',points-center)]:
        q=np.einsum('ni,nij,nj->n',residual,precision,residual)
        out[name]=dict(median_squared_mahalanobis=float(np.median(q)),coverage={str(p):ratio(int((q<=threshold).sum()),len(q)) for p,threshold in [(0.68,2.27886856637673),(.95,5.99146454710798)]})
    out['scope']='Full correlated ICRS PM covariance, nominal chi-square2 ellipses; conditional W0 plus reference, no certification of missing systematic terms. Centering is diagnostic, not a correction.'
    return out


def metrics(rows,methods):
    selected=[r for r in rows if set(methods)<=set(r['values'])]
    pairs=tuple((methods[0],other) for other in methods[1:])
    out=comparison([{m:r['values'][m] for m in methods} for r in selected],SETTINGS,pairs)
    if len(selected)<5:return out
    points=np.array([r['values'][methods[0]][0] for r in selected]);errors=np.array([r['values'][methods[0]][1] for r in selected]);center=np.median(points,axis=0)
    out['primary_centered_coverage_diagnostic']={str(k):[ratio(int(n),len(selected)) for n in (abs(points-center)<=k*errors).sum(axis=0)] for k in (1,2)}
    out['primary_raw_coverage_count_intervals']={str(k):[ratio(int(n),len(selected)) for n in (abs(points)<=k*errors).sum(axis=0)] for k in (1,2)}
    if all('pm_covariance' in r for r in selected):out['primary_2D_error_ellipse']=ellipse_coverage(points,[r['pm_covariance'] for r in selected])
    out['coverage_scope']='Axis intervals condition on these objects and measured shared groups; not independent-sky intervals. Centering is diagnostic only, not a catalogue correction.'
    return out


def in_magnitude_bin(row,low):
    value=row.get('magnitude')
    return value is not None and np.isfinite(value) and low<=value<low+1


def fixed_angle_readout(records):
    output={}
    for low in range(0,70,10):
        part=[r for r in records if r['chart_angle_deg'] is not None and low<=r['chart_angle_deg']<low+10]
        computed=[r for r in part if r['status']=='computed'];out=dict(frozen_denominator=len(part),computed=len(computed),status=dict(Counter(r['status'] for r in part)))
        if len(computed)>=5:
            old=np.array([r['old_values']['blind_all'][1] for r in computed]);ratios={name:np.array([r[key][1] for r in computed])/old for name,key in [('B1_over_old','B1_values'),('new_over_old','new_values')]}
            rng=np.random.default_rng(2227);indices=rng.integers(0,len(computed),(256,len(computed)))
            out['error_ratio']={name:dict(median=np.median(value,axis=0),median_source_sampling95=np.quantile(np.median(value[indices],axis=1),[.025,.975],axis=0).T) for name,value in ratios.items()}
            points=[dict(r,values=dict(new=r['new_values'],old=r['old_values']['blind_all'],B1=r['B1_values'])) for r in computed]
            out['same_identity_quality']=metrics(points,('new','old','B1'))
        output[str(low)]=out
    return output


def stratified(rows):
    result={}
    for low in (20,21,22):
        base=[r for r in rows if in_magnitude_bin(r,low)];groups={'all':base}
        for tag,predicate in [('one_run',lambda r:r['sdss_runs']==1),('multi_run',lambda r:r['sdss_runs']>1),
            ('nsc_span_lt2',lambda r:r['nsc_span']<2),('nsc_span_ge2',lambda r:r['nsc_span']>=2),
            ('night_lt5',lambda r:r['nsc_nights']<5),('night_ge5',lambda r:r['nsc_nights']>=5),
            ('season_lt3',lambda r:r['nsc_seasons']<3),('high_leverage',lambda r:r['leverage']>=.8),
            ('NSC_years_ge3',lambda r:r.get('nsc_years') is not None and r['nsc_years']>=3),
            ('SDSS_years_ge2',lambda r:r.get('sdss_years') is not None and r['sdss_years']>=2),
            ('joint_span_ge12',lambda r:r['joint_span']>=12),('far_chart_ge40',lambda r:r['chart_angle_deg'] is not None and r['chart_angle_deg']>=40)]:groups[tag]=[r for r in base if predicate(r)]
        for angle in range(0,70,10):groups[f'chart_angle_{angle}_{angle+10}']=[r for r in base if r['chart_angle_deg'] is not None and angle<=r['chart_angle_deg']<angle+10]
        for key in ('kind','instruments','window','bands'):
            for value in sorted({r[key] for r in base}):groups[key+'__'+value]=[r for r in base if r[key]==value]
        result[str(low)]={}
        for tag,part in groups.items():
            result[str(low)][tag]=dict(spectral_denominator=len(part),all_new=metrics(part,('new_all',)),
                strict_published_common=metrics(part,('new_all','Xiao','NSC')),
                strict_physical_r_common=metrics(part,('new_all','new_physical_r')),
                strict_old_catalogue_common=metrics(part,('new_all','old_all','Xiao','NSC')))
        result[str(low)]['all']['reference_common']=reference_common(base)
        common=[r for r in base if {'new_all','Xiao','NSC'}<=r['values'].keys()]
        observed=[dict(r,motion={m:r['values'][m][0] for m in ('new_all','Xiao','NSC')}) for r in common]
        result[str(low)]['all']['physical_observation_sensitivity']=estimate(observed,('new_all','Xiao','NSC'),2226,candidate='new_all')
    return result


def run(root,workers,frozen=None):
    import duckdb
    plan=read(root/'registration/private_full_coordinate_plan.json');dest=root/'quality';dest.mkdir(exist_ok=True)
    db=duckdb.connect(str(root/'catalogue/catalogue.duckdb'),read_only=True);quarantined=set(db.execute('SELECT src FROM not_adopted').fetchnumpy()['src']);db.close()
    old=old_index(root.parent/'v1.9/batch/science' if frozen is None else frozen)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for _ in pool.map(source_pool,[(i,e,str(root),old,quarantined) for i,e in enumerate(plan['cores'])]):pass
    fixed=[];blind=[];pool_exits=Counter()
    for e in plan['cores']:
        part=read(dest/'cores'/e['core']/'private_qso_pool.json.gz');fixed.extend(part['fixed']);blind.extend(part['blind']);pool_exits.update(read(dest/'cores'/e['core']/'private_pool_receipt.json')['exits'])
    # Exact old label identity is the frozen baseline's convention; preserve its
    # denominator, then separately report near-duplicate sensitivity if needed.
    counts=Counter(tuple(r['identity']) for r in fixed);fixed_unique=[r for r in fixed if counts[tuple(r['identity'])]==1]
    new,duplicate=unique_positions(blind)
    old_labels=defaultdict(list)
    for r in fixed_unique:old_labels[r['identity'][0]].append(r)
    for origin,prior in old_labels.items():
        tree=cKDTree(coordinates(*np.array([r['identity'][1:] for r in prior]).T))
        for r in new:
            if r['identity'][0]!=origin:continue
            near=tree.query_ball_point(coordinates(*r['identity'][1:])[0],2*np.sin(.036/3600000*np.pi/180/2))
            if len(near)==1:r['values']['old_all']=prior[near[0]]['old_values']['blind_all']
    fixed_report={}
    for low in (20,21,22):
        before=[r for r in fixed_unique if in_magnitude_bin(r,low)];computed=[r for r in before if r['status']=='computed']
        points=[dict(r,values=dict(old=r['old_values']['blind_all'],new=r['new_values'],B1=r['B1_values'])) for r in computed]
        fixed_report[str(low)]=dict(frozen_denominator=len(before),status=dict(Counter(r['status'] for r in before)),
            local_support_exits=sum(not r['local_support_pass'] for r in computed),formal_error_exits=sum(not r['formal_error_pass'] for r in computed),
            residual_exits=sum(not r['residual_pass'] for r in computed),new_reference_common_on_fixed_identities=reference_common(computed),by_chart_angle=fixed_angle_readout(before),all_computed_same_members=metrics(points,('new','old','B1')),
            scope='New quality failures stay in same-member evaluation. Only unavailable new fits leave computed comparison, with explicit original denominator/status.')
    spatial=[]
    for low in (20,21,22):
        for pixel in sorted({r['pixel64'] for r in new if in_magnitude_bin(r,low)}):
            part=[r for r in new if r['pixel64']==pixel and in_magnitude_bin(r,low)]
            spatial.append(dict(pixel64=pixel,r_bin=low,spectral_denominator=len(part),comparison=metrics(part,('new_all','Xiao','NSC'))))
    report=dict(frozen_old_all_magnitude_identities=len(fixed_unique),frozen_old_all_identity_status=dict(Counter(r['status'] for r in fixed_unique)),
        frozen_old_unbinned_magnitude=sum(not any(in_magnitude_bin(r,low) for low in (20,21,22)) for r in fixed_unique),
        new_unbinned_magnitude=sum(not any(in_magnitude_bin(r,low) for low in (20,21,22)) for r in new),source_pool_exits=dict(pool_exits),fixed_old_QSO=fixed_report,new_final_blind=stratified(new),new_unique_spectral_sources=len(new),new_near_duplicate_claims_excluded=duplicate,
        old_exact_duplicate_claims_excluded=len(fixed)-len(fixed_unique),
        spatial_nside64=spatial,slurm_job=os.environ['SLURM_JOB_ID'],
        scope='Frozen old-source identities before quality selection and newly generated global-owned blind catalogue are independent evaluation denominators. New catalogue/source members freeze before spectrum lookup. Xiao DESI ground only and official NSC, no faint Gaia PM ranking. Source/object bootstrap describes conditional object sampling; common-frame and physical-observation sensitivity stay separate. Spatial pixels with <5 strict objects publish counts only, not precision/zero certification.')
    check_aggregate(report);save_json(dest/'v22_global_QSO_quality_summary.json',report)
    write_rows(dest/'private_final_blind_QSO.json.gz',new)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--workers',type=int,default=4);p.add_argument('--frozen',type=Path);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use Slurm')
    run(a.root,a.workers,a.frozen)
