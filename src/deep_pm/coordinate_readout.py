"""Public fixed-identity scientific readout of the three geometry contrasts."""
import argparse
from collections import defaultdict
import os
from pathlib import Path

import healpy as hp
import numpy as np

from .blind_catalogue_quality import read
from .io import digest,save_json
from .public_summary import check_aggregate
from .usability_quality import split_identity


def interval(success,total):
    if not total:return None
    p=success/total;z=1.959963984540054;d=1+z*z/total;c=(p+z*z/(2*total))/d
    h=z*np.sqrt(p*(1-p)/total+z*z/(4*total*total))/d
    return [c-h,c+h]


def stats(points,errors,training_center=None):
    v=np.asarray(points,float);e=np.asarray(errors,float);n=len(v);result=dict(n=n)
    if n<3:return result
    median=np.median(v,axis=0);z=v/e;cz=(v-median)/e
    result.update(median=median,nmad=1.4826*np.median(abs(v-median),axis=0),absolute_radial_p68=float(np.quantile(np.linalg.norm(v,axis=1),.68)),
        centered_radial_p68=float(np.quantile(np.linalg.norm(v-median,axis=1),.68)),
        standardized_nmad=1.4826*np.median(abs(z-np.median(z,axis=0)),axis=0),
        error_p50=np.median(e,axis=0),tail_radial_gt10_fraction=float(np.mean(np.linalg.norm(v,axis=1)>10)),
        tail_radial_gt20_fraction=float(np.mean(np.linalg.norm(v,axis=1)>20)))
    for factor in (1,2):
        for name,x in [('raw',z),('evaluation_centered_descriptive',cz)]+([('training_centered',(v-training_center)/e)] if training_center is not None else []):
            count=np.sum(abs(x)<=factor,axis=0)
            result[f'{name}_coverage_{factor}sigma']=dict(successes=count,denominator=n,fraction=count/n,
                wilson95=[interval(c,n) for c in count])
    rng=np.random.default_rng(210621);index=rng.integers(0,n,(400,n))
    result['median_object_conditional_interval95']=np.quantile(np.median(v[index],axis=1),[.025,.975],axis=0).T
    result['P68_object_conditional_interval95']=np.quantile(np.quantile(np.linalg.norm(v[index],axis=2),.68,axis=1),[.025,.975])
    return result


def compare(records,methods):
    rows=[r for r in records if all(m in r['values'] for m in methods)]
    out=dict(n=len(rows),methods={})
    train=[r for r in rows if split_identity(r['identity'])<3]
    for method in methods:
        center=np.median([r['values'][method][0] for r in train],axis=0) if len(train)>=5 else None
        out['methods'][method]=stats([r['values'][method][0] for r in rows],[r['values'][method][1] for r in rows],center)
    out['training_center_identities']=len(train)
    return out


def weighted_quantile(value,weight,q=.68):
    order=np.argsort(value);v=np.asarray(value)[order];w=np.asarray(weight)[order]
    return float(np.interp(q*np.sum(w),np.cumsum(w),v))


def grouped_sensitivity(records,methods):
    if len(records)<5:return dict(n=len(records))
    labels=sorted({e for r in records for e in r['exposures'] if e.startswith('SDSS:')})
    index={k:j for j,k in enumerate(labels)};rng=np.random.default_rng(210621);draws=rng.exponential(size=(256,len(labels)))
    weights=np.asarray([draws[:,[index[e] for e in r['exposures'] if e.startswith('SDSS:')]].mean(axis=1) for r in records]).T
    out=dict(n=len(records),physical_SDSS_runs=len(labels),scope='SDSS run exponential reweighting sensitivity, not a confidence interval for all NSC correlations. Each multi-run source receives its mean run multiplier. Object bootstrap is conditional and separate.')
    for m in methods:
        radial=np.linalg.norm([r['values'][m][0] for r in records],axis=1)
        out[m]=dict(P68_SDSS_run_sensitivity95=np.quantile([weighted_quantile(radial,w) for w in weights],[.025,.975]))
    return out


def run(registration,fixed,association,output):
    output.mkdir(parents=True,exist_ok=True);plan=read(registration);fixedrows=[];angular=[];corestats=[];common_refs=defaultdict(list)
    for row in plan['selected']:
        core=row['core'];f=read(fixed/core/'private_coordinate_qso.json');g=read(association/core/'private_geometry_qso.json')
        summary=read(fixed/core/'v21_coordinate_core_summary.json');asc=read(association/core/'v21_coordinate_association_core_summary.json')
        fixedrows.extend(f);corestats.append(dict(core=core,theta_deg=row['center_theta_deg'],fixed=summary,association=asc))
        index={tuple(r['identity']):r for r in f}
        for r in g:
            old=index.get(tuple(r['identity']))
            if old is None:continue
            r.update(exposures=old['exposures'],nsc_span=old['nsc_span'],theta_deg=old['theta_deg'])
            angular.append(r)
        # Mean perturbations must be summed synchronously within a graph.
        states=read(fixed/core/'private_coordinate_states.json.gz');original=read(Path(row['fit'])/'private_standard_motion.json.gz')
        byslot={r['fixed_source_slot']:r for r in f}
        for slot,r in byslot.items():
            if not 21<=r['magnitude']<23:continue
            state=states[slot];j=np.asarray(original[slot]['state_jacobian'])[[1,3]]
            b1=np.asarray(state['b1_reference_state_draws'])@j.T
            b2=np.asarray(state['b2_reference_state_draws'])@np.asarray(state['b2']['state_jacobian'])[[1,3]].T
            common_refs[(r['graph_sha256'],int(r['magnitude']))].append((tuple(r['identity']),b1,b2))
    # Overlap halos can repeat a physical target. A deterministic one-row choice
    # by registered core name is used before comparing any method values.
    def unique(rows):
        out={}
        for r in sorted(rows,key=lambda r:r['core']):out.setdefault(tuple(r['identity']),r)
        return list(out.values())
    before=len(fixedrows);fixedrows=unique(fixedrows);angular=unique(angular)
    methods=['old','B1','B1_sandwich','B2'];angularmethods=['old','B2_selected','B3_spherical']
    scientific={};spatial=[]
    for lo in (21,22):
        sample=[r for r in fixedrows if lo<=r['magnitude']<lo+1];a=[r for r in angular if lo<=r['magnitude']<lo+1]
        scientific[str(lo)]=dict(fixed_members=compare(sample,methods),all_three_identity_common=compare(a,angularmethods),
            external_strict_common=compare([r for r in sample if all(m in r['values'] for m in ('NSC','Xiao'))],['old','B2','Xiao','NSC']),
            physical_group_sensitivity=grouped_sensitivity(sample,['old','B2']),angle_bins={},sampling={},window={})
        for low,high in [(0,10),(10,20),(20,30),(30,40),(40,50),(50,70)]:
            rows=[r for r in sample if low<=r['theta_deg']<high]
            scientific[str(lo)]['angle_bins'][f'{low}_{high}']=compare(rows,methods)
        for name,predicate in [('single_SDSS_run',lambda r:len({e for e in r['exposures'] if e.startswith('SDSS:')})==1),
                               ('multi_SDSS_run',lambda r:len({e for e in r['exposures'] if e.startswith('SDSS:')})>1),
                               ('NSC_shorter_2yr',lambda r:r['nsc_span'] is not None and r['nsc_span']<2),
                               ('NSC_atleast_2yr',lambda r:r['nsc_span'] is not None and r['nsc_span']>=2)]:
            scientific[str(lo)]['sampling'][name]=compare([r for r in sample if predicate(r)],methods)
        for window in sorted({r['window'] for r in sample}):scientific[str(lo)]['window'][window]=compare([r for r in sample if r['window']==window],methods)
        groups=defaultdict(list)
        for r in sample:
            _,ra,dec=r['identity'];pixel=int(hp.ang2pix(32,ra,dec,lonlat=True));groups[pixel].append(r)
        for pixel,rows in sorted(groups.items()):
            spatial.append(dict(pixel_ring_nside32=pixel,magnitude_low=lo,median_theta_deg=float(np.median([r['theta_deg'] for r in rows])),**compare(rows,methods)))
    synchronous={}
    for lo in (21,22):
        n=0;cov1=np.zeros((2,2));cov2=np.zeros((2,2));seen=set()
        for (graph,mag),group in sorted(common_refs.items()):
            if mag!=lo:continue
            use=[]
            for identity,b1,b2 in group:
                if identity not in seen:seen.add(identity);use.append((b1,b2))
            if not use:continue
            n+=len(use);cov1+=np.cov(np.sum([v[0] for v in use],axis=0),rowvar=False);cov2+=np.cov(np.sum([v[1] for v in use],axis=0),rowvar=False)
        synchronous[str(lo)]=dict(n=n,B1_equal_mean_reference_covariance=cov1/max(n*n,1),B2_equal_mean_reference_covariance=cov2/max(n*n,1),
            scope='Synchronous48 draws within actual graph; graph domains independent. Sum response before covariance. Conditional on frozen C0/bootstrap, not total unknown spatial systematics.')
    report=dict(plan_sha256=digest(registration),fixed_identity_count=len(fixedrows),duplicate_halo_labels_removed=before-len(fixedrows),
        cores=corestats,scientific=scientific,same_pixel_statistics=spatial,synchronous_reference_mean=synchronous,
        slurm_job=os.environ['SLURM_JOB_ID'],scope='Geometry-selected16-core diagnostic, no representative all-sky claim. Fixed-member B1/B2 uses original labels; angular selection comparison keeps all-three actually unique physical QSO identities. Identical magnitude/sky/sampling definitions within comparisons. Wilson intervals for proportions; object bootstrap conditional; SDSS-run sensitivity and synchronous reference uncertainty separate. Original graph and source selection frozen; B3 reference-input changes not yet re-solved globally.')
    check_aggregate(report);save_json(output/'v21_scientific_comparison_summary.json',report)
    print('COORDINATE_SCIENCE_READOUT',len(fixedrows),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('registration','fixed','association','output'):p.add_argument(n,type=Path)
    a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use Slurm resources')
    run(**vars(a))
