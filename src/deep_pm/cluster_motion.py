"""Fixed-member stellar diagnostics and physical-group slope responses."""
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from .__main__ import tomllib
from .allband_qso import evaluated_fit
from .astrometry import years
from .blind_catalogue_quality import read
from .catalogue_bias_probe import galaxy_controls
from .continuous_reference import PRECISION,read_offsets,reference_frame,verify_directory
from .io import digest
from .pilot_production import quality_detections
from .precision_v03 import calibration_groups
from .selection_semantics import morphology_state


def describe(records,paths,specs):
    grouped=defaultdict(list)
    for origin,slot,row in records:grouped[origin].append((slot,row))
    result={}
    config=tomllib.loads(PRECISION.read_text())
    settings=tomllib.loads((Path(__file__).parents[2]/'configs/spatial-v08.toml').read_text())
    for origin,chosen in grouped.items():
        folder=Path(paths[origin]);source=(folder/'source_receipt.json').resolve().parent
        graph=(folder/'reference_receipt.json').resolve().parent/'private_c0.npz'
        center=reference_frame(graph);offsets=read_offsets(graph);graph_sha=digest(graph)
        verify_directory(source,('detections.csv',))
        data,xy,time,sigma,keep=quality_detections(source,center,config,settings['production'])
        lookup=pd.Index(data.detection_id);keys=calibration_groups(data);nsc=data.survey.eq('NSC').to_numpy()
        ref=(graph.parent/(origin+'_reference_receipt.json')).resolve().parent
        prediction={};reference_status='not_available'
        if (ref/'reference.csv.gz').exists():
            prediction,_,_=galaxy_controls(ref,center,offsets,specs[origin],data)
            reference_status='independent_held_galaxies'
        for slot,row in chosen:
            ix=lookup.get_indexer(row['detection_ids'])
            if (ix<0).any():raise ValueError('Frozen stellar member is absent from original input')
            t=time[ix];weight=np.asarray(row['robust_weight'])[:,None]/sigma[ix]**2
            design=np.column_stack([np.ones(len(t)),t-row['reference_epoch']])
            independent=np.asarray(row['independent_tangent_covariance']);jac=np.asarray(row['state_jacobian'])
            h=np.zeros((4,len(ix),2))
            for axis in range(2):h[2*axis:2*axis+2,:,axis]=independent[2*axis:2*axis+2,2*axis:2*axis+2]@(design.T*weight[:,axis])
            k=np.einsum('ap,pnd->and',jac[[1,3]],h)
            draws=np.stack([offsets[g][1] for g in keys[ix]],axis=1)
            reference_draws=np.einsum('ani,dni->da',k,draws)
            expected=(jac@np.asarray(row['reference_tangent_covariance'])@jac.T)[np.ix_([1,3],[1,3])]
            np.testing.assert_allclose(np.cov(reference_draws,rowvar=False),expected,atol=2.e-7,rtol=2.e-8)
            responses={graph_sha+':'+g:k[:,keys[ix]==g,:].sum(axis=1) for g in sorted(set(keys[ix]))}
            early=~nsc[ix];available=all(g in prediction for g in keys[ix][early]) and bool(early.any())
            galaxy_shift=sum(k[:,j,:]@prediction[keys[ix[j]]] for j in np.flatnonzero(early)) if available else None
            variants={}
            for name,use in [('physical_r_members',ix[data.band.iloc[ix].eq('r').to_numpy()]),('nsc_only_members',ix[nsc[ix]])]:
                late=use[nsc[use]]
                if len(use)>=3 and len(late)>=2 and np.ptp(time[late])>=.6 and np.ptp(time[use])>0:
                    fit=evaluated_fit(use,data,xy,time,sigma,keys,offsets,center,row['reference_epoch'])
                    variants[name]=dict(motion=fit['pm_icrs_masyr'],error=fit['pm_error_masyr'],used=len(use),span=np.ptp(time[use]))
            colors=data.color.iloc[ix[early]].to_numpy(float) if 'color' in data else np.array([])
            colors=colors[np.isfinite(colors)&(colors>-1)&(colors<4)]
            result[(origin,slot)]=dict(row=row,variants=variants,response=responses,reference_draws=reference_draws,
                independent_icrs=(jac@independent@jac.T)[np.ix_([1,3],[1,3])],graph_sha256=graph_sha,
                galaxy_predicted_pm=galaxy_shift,galaxy_reference_status=reference_status,
                sdss_g_minus_i=float(np.median(colors)) if len(colors) else None,
                adopted_nsc_morphology=morphology_state(data.morphology.iloc[ix[nsc[ix]]]),
                sdss_runs=sorted(set(data.exposure_id.iloc[ix[early]])))
        print('CLUSTER_CORE_DESCRIBED',origin,len(chosen),flush=True)
    return result


def response_features(records):
    groups=sorted({g for r in records for g in r['response']});lookup={g:i for i,g in enumerate(groups)}
    rr=[];cc=[];vv=[]
    for j,row in enumerate(records):
        for group,values in row['response'].items():
            flat=np.asarray(values).ravel();rr.extend([j]*4);cc.extend(lookup[group]*4+np.arange(4));vv.extend(flat)
    matrix=csr_matrix((vv,(rr,cc)),shape=(len(records),len(groups)*4))
    norm=np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
    return matrix.multiply(1/np.maximum(norm,1.e-30)[:,None]).tocsr(),norm


def support_predictions(records,training_indices):
    matrix,norm=response_features(records)
    train=np.asarray(training_indices,int)
    scores=(matrix@matrix[train].T).toarray()
    ratio=norm[:,None]/np.maximum(norm[train],1.e-30)
    scores[(ratio<.7)|(ratio>1.3)]=-1
    output={}
    for j,values in enumerate(scores):
        eligible=np.flatnonzero(values>=.8)
        eligible=eligible[np.argsort(values[eligible],kind='stable')[-10:]]
        if len(eligible)<8:continue
        chosen=train[eligible]
        output[j]=dict(predicted_zero=np.median([records[k]['row']['pm_icrs_masyr'] for k in chosen],axis=0),
            training_neighbors=chosen.tolist(),minimum_response_cosine=float(values[eligible].min()),
            relative_response_amplitude=np.asarray(ratio[j,eligible]))
    return output


def mean(values,weights=None,reference_motion=None):
    out=dict(n=len(values))
    if len(values)<3:return out
    motion=np.asarray([r['row']['pm_icrs_masyr'] for r in values])
    error=np.asarray([r['row']['pm_error_masyr'] for r in values])
    if weights is None:weights=1/np.mean(error**2,axis=1)
    weights=np.asarray(weights,float);weights/=weights.sum()
    independent=sum(a*a*np.asarray(r['independent_icrs'],float) for a,r in zip(weights,values))
    graphs=defaultdict(list)
    for a,r in zip(weights,values):graphs[r['graph_sha256']].append(a*np.asarray(r['reference_draws'],float))
    common=sum(np.cov(np.sum(v,axis=0),rowvar=False) for v in graphs.values())
    out.update(weighted_mean=weights@motion,median=np.median(motion,axis=0),
        independent_covariance=independent,synchronous_reference_covariance=common,
        independent_standard_error=np.sqrt(np.diag(independent)),reference_common_standard_error=np.sqrt(np.diag(common)),
        reference_domains=len(graphs),effective_independent_weights=1/(weights@weights))
    if reference_motion is not None:out['offset_from_bright_reference']=out['weighted_mean']-np.asarray(reference_motion)
    return out
