"""External HACKS identities: freeze quality and positions before PM residuals."""
import argparse
from collections import Counter,defaultdict
from hashlib import sha256
from pathlib import Path
import os,re
import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.spatial import cKDTree
from scipy.stats import chi2
from .astrometry import project,unproject
from .blind_catalogue_quality import read
from .coordinate_metric import AngularTree,angular_distance
from .dark_precision import query_unique
from .full_coordinate_sources import write_rows
from .io import digest,save_json
from .public_summary import check_aggregate
from .validation import ratio

RULES=dict(max_pm_error=.5,max_reduced_chi2=1.5,min_PM_NU=5,min_PM_used_fraction=.8,min_baseline_years=5,
    max_photo_rms=.15,min_qfit=.9,max_abs_radxs=.1,max_crowding_o=1.,min_photo_used_fraction=.75,min_sky_SNR=3.5,
    match_mas=700.,margin_mas=300.,bright_neighbor_radius_mas=1200.,neighbor_limit_mag=2.5,contamination_radius_mas=1500.,max_neighbor_flux_fraction=.1,
    magnitude_edges=[20,21,22,23],min_publish_count=5,bootstrap_draws=512,seed=230027)
ZERO={'ngc5466':dict(value=[-5.371,-.800],error=[.030,.031]),'ngc5272':dict(value=[-.261,-2.674],error=[.055,.040]),'ngc5904':dict(value=[4.079,-9.876],error=[.036,.036])}


def comments(file):
    result=[]
    with file.open() as stream:
        for line in stream:
            if not line.startswith('#'):break
            result.append(line)
    return ''.join(result)


def hst_parent(folder,name,center):
    pmfile=next(folder.glob('*_pm-catalog.fits'));header=comments(pmfile.with_suffix('.txt'))
    epoch=float(re.search(r'reference epoch \(yr\)\s*~\s*([0-9.]+)',header).group(1))
    match=re.search(r'zero-point\s*=\s*\(([-.0-9]+)\+/-([.0-9]+),([-.0-9]+)\+/-([.0-9]+)\)',header)
    measured=np.array([float(match[1]),float(match[3])]);stat=np.array([float(match[2]),float(match[4])])
    np.testing.assert_allclose(measured,ZERO[name]['value'],atol=.0001,rtol=0)
    with fits.open(pmfile,memmap=False) as h:pm={c:np.array(h[1].data[c]) for c in h[1].columns.names}
    n=len(pm['ID']);good=np.isfinite(np.column_stack([pm[c] for c in ['RA','DEC','PMRAC','PMDECC','EPMRAC','EPMDECC','CHI2RX','CHI2RY']])).all(axis=1)
    good&=(pm['EPMRAC']>0)&(pm['EPMRAC']<=.5)&(pm['EPMDECC']>0)&(pm['EPMDECC']<=.5)&(pm['CHI2RX']>=0)&(pm['CHI2RY']>=0)&(pm['CHI2RX']<=1.5)&(pm['CHI2RY']<=1.5)&(pm['NU']>=5)&(pm['NU']>=.8*pm['NF'])&(pm['NF']>0)&(pm['TIME']>=5)&(pm['CORRFLAG']==1)
    pm_good=good.copy();epochgood=defaultdict(list);phot606=[];phot814=[];files={pmfile.name:digest(pmfile)}
    for file in sorted(folder.glob('*_phot-catalog-*.fits')):
        text=comments(file.with_suffix('.txt'));zp=float(re.search(r'VEGA magnitude zero-point\s*=\s*([0-9.]+)',text).group(1))
        with fits.open(file,memmap=False) as h:t={c:np.array(h[1].data[c]) for c in h[1].columns.names}
        if not np.array_equal(t['ID'],pm['ID']):raise ValueError('HST photometry identity/order differs from PM catalogue')
        mag=t['MAG'];ok=(mag>0)&(mag<35)&(t['RMSMAG']>0)&(t['RMSMAG']<=.15)&(t['QFIT']>=.9)&(abs(t['RADXS'])<=.1)&(t['O']<1)&(t['NF']>0)&(t['NU']>=.75*t['NF'])&(t['SATFLAG']<9)&(t['RMSSKY']>0)
        ok&=10**np.clip(-.4*(mag-zp),-100,100)>=3.5*t['RMSSKY']
        tag=re.search(r'phot-catalog-(ep\d+)',file.name).group(1);epochgood[tag].append(ok)
        if '_f606w_' in file.name:phot606.append(np.where((mag>0)&(mag<35)&(t['SATFLAG']<9),mag,np.nan))
        if '_f814w_' in file.name:phot814.append(np.where((mag>0)&(mag<35)&(t['SATFLAG']<9),mag,np.nan))
        files[file.name]=digest(file)
    epochs=np.array([np.sum(v,axis=0)>=min(2,len(v)) for v in epochgood.values()]);photo_good=epochs.sum(axis=0)>=2;good&=photo_good
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        m606=np.nanmedian(phot606,axis=0) if phot606 else np.full(n,np.nan);m814=np.nanmedian(phot814,axis=0) if phot814 else np.full(n,np.nan)
    rel=np.column_stack([pm['PMRAC'],pm['PMDECC']]);absolute=rel+measured
    xy=project(pm['RA'],pm['DEC'],center)
    # The HST field is < a few arcmin. Explicit native->chart derivative keeps
    # external epoch propagation in the same physical angular metric.
    from .coordinate_metric import forward_jacobian
    velocity=np.einsum('nij,nj->ni',forward_jacobian(pm['RA'],pm['DEC'],center),np.where(np.isfinite(absolute)&(abs(absolute)<1e4),absolute,measured))
    positions=unproject(xy+(2015.5-epoch)*velocity,center);xy2015=project(*positions.T,center)
    tree=AngularTree(xy2015,center);near=tree.query_ball_point(xy2015,1500);isolated=np.zeros(n,bool);nearest=np.full(n,np.inf);fraction=np.full(n,np.nan)
    for i,neighbors in enumerate(near):
        others=np.array([j for j in neighbors if j!=i],int)
        if not np.isfinite(m606[i]):continue
        valid=others[np.isfinite(m606[others])]
        brighter=valid[m606[valid]<=m606[i]+2.5];db=angular_distance(xy2015[brighter],xy2015[i],center)
        fraction[i]=np.sum(10**(-.4*(m606[valid]-m606[i])))
        if len(db):nearest[i]=db.min()
        isolated[i]=(not len(db) or db.min()>=1200) and fraction[i]<=.1
    return dict(name=name,epoch=epoch,pm=pm,relative=rel,absolute=absolute,position2015=positions,xy2015=xy2015,center=center,pm_good=pm_good,photo_good=photo_good,good=good,isolated=isolated,m606=m606,m814=m814,neighbor_fraction=fraction,nearest_bright_neighbor_mas=nearest,zero=measured,zero_stat_error=stat,zero_total_error=np.asarray(ZERO[name]['error']),files=files)


def prepare(base,root):
    import duckdb
    output=root/'hst';output.mkdir(exist_ok=True);registration=read(root/'preflight/private_HST_geometric_registration.json')
    db=duckdb.connect(str(base/'v2.2/catalogue/catalogue.duckdb'),read_only=True);db.execute("SET threads=2");db.execute("SET memory_limit='8GB'")
    allparents=[];allmatches=[];counts=[];sourceproofs={};details=[]
    for field in registration['selected']:
        name=field['field'];center=field['center_deg'];p=hst_parent(root/'preflight/HACKS'/name,name,center);n=len(p['good'])
        r,d=p['position2015'].T;ground=db.execute('SELECT src,core,slot,ra_deg,dec_deg,magnitude,sdss_runs,nsc_span,nsc_nights,nsc_seasons,bands,instruments,kind FROM accepted WHERE ra_deg BETWEEN ? AND ? AND dec_deg BETWEEN ? AND ?', [float(r.min()-.002),float(r.max()+.002),float(d.min()-.002),float(d.max()+.002)]).fetchdf()
        gxy=project(ground.ra_deg,ground.dec_deg,center);tree=AngularTree(gxy,center);valid,nearest=query_unique(tree,p['xy2015'],700,300) if len(ground) else (np.zeros(n,bool),np.zeros(n,int))
        htree=AngularTree(p['xy2015'],center)
        if len(ground):
            othergood,back=query_unique(htree,gxy,700,300)
            valid&=np.array([j<len(ground) and othergood[j] and back[j]==i for i,j in enumerate(nearest)])
        for i in range(n):
            allparents.append(dict(field=name,HST_ID=int(p['pm']['ID'][i]),position2015=p['position2015'][i],relative_pm=p['relative'][i].tolist() if np.isfinite(p['relative'][i]).all() else None,
                HST_error=[float(p['pm'][c][i]) if np.isfinite(p['pm'][c][i]) else None for c in ['EPMRAC','EPMDECC']],PM_quality=bool(p['pm_good'][i]),photometric_quality=bool(p['photo_good'][i]),quality=bool(p['good'][i]),isolated=bool(p['isolated'][i]),F606W=float(p['m606'][i]) if np.isfinite(p['m606'][i]) else None,F814W=float(p['m814'][i]) if np.isfinite(p['m814'][i]) else None,
                matched_v22=bool(valid[i]),ground_src=int(ground.iloc[nearest[i]].src) if valid[i] else None))
        for i in np.flatnonzero(p['good']&valid):
            g=ground.iloc[nearest[i]]
            allmatches.append(dict(field=name,HST_ID=int(p['pm']['ID'][i]),ground_src=int(g.src),core=str(g.core),slot=int(g.slot),magnitude=float(g.magnitude),magnitude_source='frozen_v22_NSCr_or_SDSSr',
                isolated=bool(p['isolated'][i]),F606W=float(p['m606'][i]) if np.isfinite(p['m606'][i]) else None,F814W=float(p['m814'][i]) if np.isfinite(p['m814'][i]) else None,
                relative_pm=p['relative'][i],HST_error=[p['pm']['EPMRAC'][i],p['pm']['EPMDECC'][i]],HST_position2015=p['position2015'][i],sdss_runs=int(g.sdss_runs),nsc_span=float(g.nsc_span),nsc_nights=int(g.nsc_nights),nsc_seasons=int(g.nsc_seasons),kind=str(g.kind),bands=str(g.bands),instruments=str(g.instruments)))
        counts.append(dict(field=name,HST_parent=n,pm_quality=int(p['pm_good'].sum()),photometric_quality=int(p['photo_good'].sum()),combined_quality=int(p['good'].sum()),isolated_quality=int((p['good']&p['isolated']).sum()),all_position_unique_v22=int(valid.sum()),quality_unique_v22=int((p['good']&valid).sum()),isolated_quality_unique_v22=int((p['good']&p['isolated']&valid).sum()),ground_sources_in_HST_bounding_box=len(ground)))
        details.append(dict(field=name,reference_epoch=p['epoch'],zero_point=p['zero'],header_statistical_error=p['zero_stat_error'],paper_total_shared_error=p['zero_total_error'],HST_input_sha256=p['files']))
        print('HST_FROZEN_POSITION_MATCH',name,counts[-1],flush=True)
    db.close()
    for core in sorted({r['core'] for r in allmatches}):sourceproofs[core]=digest(base/'v2.2/sources'/core/'v22_source_core_summary.json')
    write_rows(output/'private_HST_parent.json.gz',allparents);write_rows(output/'private_frozen_position_matches.json.gz',allmatches)
    save_json(output/'private_HST_fields.json',details)
    frozen=dict(rules=RULES,protocol_sha256=digest(Path('docs/faint_star_protocol_v23.md')),metadata_registration_sha256=digest(root/'preflight/private_HST_geometric_registration.json'),parent_sha256=digest(output/'private_HST_parent.json.gz'),position_matches_sha256=digest(output/'private_frozen_position_matches.json.gz'),source_summary_sha256=sourceproofs,fields=details,stage='quality_and_position_only_before_ground_PM_residuals',slurm_job=os.environ['SLURM_JOB_ID'])
    save_json(output/'private_quality_freeze.json',frozen)
    out=dict(fields=counts,HST_pm_quality_rules=RULES,reference_frame=[{k:r[k] for k in ['field','reference_epoch','zero_point','header_statistical_error','paper_total_shared_error']} for r in details],freeze_sha256=digest(output/'private_quality_freeze.json'),slurm_job=os.environ['SLURM_JOB_ID'],scope='PM quality and unique identities fixed before reading ground motion residuals. HST PM only external epoch registration and validation. Photometric HST central parents separate from frozen ground outer-ring415/40. Crowding is conditional on the released HST PM catalogue; unseen companions remain possible.')
    check_aggregate(out);save_json(output/'v23_HST_frozen_overlap_summary.json',out)


def metrics(residual,covariance,common_zero):
    n=len(residual);out=dict(n=n)
    if n<5:return out
    x=np.asarray(residual);c=np.asarray(covariance);center=np.median(x,axis=0);sigma=np.sqrt(np.diagonal(c,axis1=1,axis2=2));rad=np.linalg.norm(x,axis=1);centered=x-center
    rng=np.random.default_rng(230027);boot=np.array([[np.median(x[ix],axis=0),np.quantile(np.linalg.norm(x[ix],axis=1),.68)*np.ones(2)] for ix in rng.integers(0,n,(512,n))])
    out.update(median=center,nmad=1.4826*np.median(abs(centered),axis=0),absolute_radial_P68=float(np.quantile(rad,.68)),centered_radial_P68=float(np.quantile(np.linalg.norm(centered,axis=1),.68)),median_conditional_source_interval95=np.quantile(boot[:,0],[.025,.975],axis=0).T,P68_conditional_source_interval95=np.quantile(boot[:,1,0],[.025,.975]),tail_radial_gt20=ratio(int((rad>20).sum()),n),equal_mean=x.mean(axis=0),mean_catalogue_covariance_over_n_assuming_independence=c.sum(axis=0)/n**2,HST_zero_shared_covariance=common_zero)
    for label,y in [('raw',x),('centered_diagnostic',centered)]:
        out[label]=dict(axis_1sigma=[ratio(int(k),n) for k in np.sum(abs(y)<=sigma,axis=0)],axis_2sigma=[ratio(int(k),n) for k in np.sum(abs(y)<=2*sigma,axis=0)],axis_gt3sigma=[ratio(int(k),n) for k in np.sum(abs(y)>3*sigma,axis=0)])
        z=np.einsum('ni,nij,nj->n',y,np.linalg.inv(c),y);out[label]['ellipse']={str(v):ratio(int(np.sum(z<=chi2.ppf(v,2))),n) for v in [.68,.95]}
    return out


def evaluate(base,root):
    from .stream_json import array_items
    from .dark_precision import query_unique
    from .continuous_reference import PRECISION,SPATIAL
    from .__main__ import tomllib
    from .pilot_production import quality_detections
    from .faint_covariance_v23 import decompose
    from .full_coordinate_sources import reference_context
    output=root/'hst';proof=read(output/'private_quality_freeze.json')
    assert proof['position_matches_sha256']==digest(output/'private_frozen_position_matches.json.gz')
    matches=read(output/'private_frozen_position_matches.json.gz');fields={r['field']:r for r in read(output/'private_HST_fields.json')};data_rows=[];budget=[]
    plans={e['core']:e for e in read(base/'v2.2/registration/private_full_coordinate_plan.json')['cores']}
    config=tomllib.loads(PRECISION.read_text());prod=tomllib.loads(SPATIAL.read_text())['production']
    for core in sorted({r['core'] for r in matches}):
        folder=base/'v2.2/sources'/core
        if digest(folder/'v22_source_core_summary.json')!=proof['source_summary_sha256'][core]:raise ValueError('Frozen main catalogue changed')
        wanted={r['slot'] for r in matches if r['core']==core};sources={j:r for j,r in enumerate(array_items(folder/'private_standard_motion.json.gz')) if j in wanted}
        entry=plans[core];data,xy,t,sigma,_=quality_detections(Path(entry['source']),entry['center'],config,prod);graphs={}
        external={}
        for method,path in [('NSC',Path(entry['source'])/'official_nsc.csv'),('Xiao',Path(entry['fit'])/'legacy.csv')]:
            f=pd.read_csv(path,dtype={'id':str}).rename(columns=str.lower)
            if method=='Xiao':f=f.loc[f.pm_source.astype(str).str.strip().eq('DESI')]
            external[method]=f.loc[np.isfinite(f[['ra','dec']]).all(axis=1)].drop_duplicates(subset=['ra','dec']).reset_index(drop=True)
        for m in [r for r in matches if r['core']==core]:
            row=sources[m['slot']];component=str(row['reference_component'])
            if component not in graphs:graphs[component]=reference_context(str(base/'v2.2/reference'/f'component-{int(component):04d}'/'private_c0_unfiltered.npz'))
            parts=decompose(row,data,t,sigma,graphs[component][1]);zero=np.array(fields[m['field']]['zero_point']);he=np.array(m['HST_error']);truth=np.asarray(m['relative_pm'])+zero
            covariance=np.asarray(row['covariance'])[np.ix_([1,3],[1,3])];values={'v22':np.asarray(row['pm_icrs_masyr'])};covariances={'v22':covariance+np.diag(he**2)}
            center=tuple(m['HST_position2015']);goal=np.zeros((1,2))
            for method,f in external.items():
                if f.empty:continue
                valid,near=query_unique(AngularTree(project(f.ra,f.dec,center),center),goal,700,300)
                if not valid[0]:continue
                e=f.iloc[near[0]];err=e[['pmraerr','pmdecerr'] if method=='NSC' else ['pmra_err','pmdec_err']].to_numpy(float);pm=e[['pmra','pmdec']].to_numpy(float)
                if method=='NSC' and (not np.isfinite(float(e.deltamjd)) or e.deltamjd<=0):continue
                if np.isfinite(err).all() and (err>0).all() and np.isfinite(pm).all() and abs(pm).max()<1e5:values[method]=pm;covariances[method]=np.diag(err**2+he**2)
            record=dict(m,values=values,covariance=covariances,truth_absolute=truth,zero_total_error=fields[m['field']]['paper_total_shared_error'],reference_model_sha256=row['reference_model_sha256'],reference_state_draws=parts['synchronous_reference_state_draws'],independent_v22_PM=parts['survey_terms']['SDSS']+parts['survey_terms']['NSC'],budget=parts)
            data_rows.append(record);budget.append(dict(field=m['field'],core=core,magnitude=m['magnitude'],isolated=m['isolated'],sdss_runs=m['sdss_runs'],nsc_span=m['nsc_span'],survey_terms=parts['survey_terms'],reference=parts['reference'],maximum_covariance_reconstruction_error=parts['maximum_covariance_reconstruction_error']))
        print('HST_FROZEN_EVALUATION_CORE',core,len(wanted),flush=True)
    write_rows(output/'private_HST_evaluated.json.gz',data_rows)
    results={}
    for name in fields:
        result={}
        for tag,selection in [('all_quality',lambda r:True),('isolated',lambda r:r['isolated']),('crowded_or_unknown',lambda r:not r['isolated']),('r20_21',lambda r:20<=r['magnitude']<21),('r21_22',lambda r:21<=r['magnitude']<22),('r22_23',lambda r:22<=r['magnitude']<23),('single_run',lambda r:r['sdss_runs']==1),('multi_run',lambda r:r['sdss_runs']>1),('nsc_span_lt2',lambda r:r['nsc_span']<2),('nsc_span_ge2',lambda r:r['nsc_span']>=2),('F606_21_22',lambda r:r['F606W'] is not None and 21<=r['F606W']<22),('F606_22_23',lambda r:r['F606W'] is not None and 22<=r['F606W']<23)]:
            part=[r for r in data_rows if r['field']==name and selection(r)];output_methods={};common=[r for r in part if all(k in r['values'] for k in ['v22','Xiao','NSC'])]
            for comparison,rows,methods in [('v22_available',part,['v22']),('strict_triple',common,['v22','Xiao','NSC'])]:
                entry=dict(n=len(rows),methods={})
                for method in methods:
                    if not rows:entry['methods'][method]=dict(n=0);continue
                    residual=np.array([r['values'][method]-r['truth_absolute'] for r in rows]);cov=np.array([r['covariance'][method] for r in rows]);z=np.diag(np.asarray(fields[name]['paper_total_shared_error'])**2)
                    detail=metrics(residual,cov,z)
                    if method=='v22' and len(rows)>=5:
                        independent=sum(np.asarray(r['independent_v22_PM'])+np.diag(np.asarray(r['HST_error'])**2) for r in rows)/len(rows)**2;grouped=defaultdict(list)
                        for r in rows:grouped[r['reference_model_sha256']].append(np.asarray(r['reference_state_draws'])[:,[1,3]])
                        reference=sum(np.cov(np.sum(v,axis=0)/len(rows),rowvar=False) for v in grouped.values())
                        detail['mean_independent_covariance']=independent;detail['mean_synchronous_reference_covariance']=reference;detail['mean_total_conditional_with_shared_HST_zero']=independent+reference+z
                    entry['methods'][method]=detail
                output_methods[comparison]=entry
            result[tag]=output_methods
        results[name]=result
    summary=dict(fields=results,quality_and_identity_freeze_sha256=digest(output/'private_quality_freeze.json'),matched_HST_quality_sources=len(data_rows),covariance_budget_summary=dict(n=len(budget),maximum_reconstruction_error=max([r['maximum_covariance_reconstruction_error'] for r in budget],default=0.)),slurm_job=os.environ['SLURM_JOB_ID'],
        scope='Same external physical HST identities and frozen v22. Relative scatter is centered diagnostic; absolute residual uses HST relative PM + published Gaia-anchored zero. HST-zero covariance is shared, not divided by N. Per-source HST and historical catalogues lack published cross-axis covariance here. Published historical positions lack an exact position epoch in local cache: unique0.7arcsec/margin0.3arcsec spatial counterpart evidence only, no PM-agreement matching. Cluster rotation suppressed by relative HST reference is an additional absolute-comparison limitation. Intervals are conditional object resampling, not independent spatial certification.')
    check_aggregate(summary);save_json(output/'v23_HST_validation_summary.json',summary);write_rows(output/'private_HST_covariance_budget.json.gz',budget)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','evaluate']);p.add_argument('base',type=Path);p.add_argument('root',type=Path);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use Slurm')
    (prepare if a.stage=='prepare' else evaluate)(a.base,a.root)
