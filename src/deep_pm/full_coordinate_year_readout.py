"""Whole-source sampling intervals for archived held-year position readouts.

No association or fit rerun. Original private receipts and diagnostics persist;
public intervals keep all repeated test measurements of a source together.
"""
import argparse
from collections import defaultdict
from pathlib import Path
import os
import numpy as np
from .blind_catalogue_quality import read
from .io import digest,save_json
from .public_summary import check_aggregate


def source_interval(vectors,identities,draws=256,seed=2227):
    vectors=np.asarray(vectors);_,inverse=np.unique(identities,return_inverse=True);n=int(inverse.max()+1)
    values=np.column_stack([vectors,np.linalg.norm(vectors,axis=1)]);order=np.argsort(values,axis=0);sorted_values=np.take_along_axis(values,order,axis=0)
    rng=np.random.default_rng(seed);out=[];q=np.array([.5,.5,.68])
    for _ in range(draws):
        counts=rng.multinomial(n,np.repeat(1/n,n));weight=counts[inverse];cumulative=np.cumsum(weight[order],axis=0);rank=q*(weight.sum()-1);a=np.floor(rank).astype(int);b=np.ceil(rank).astype(int)
        ia=np.array([np.searchsorted(cumulative[:,j],a[j],side='right') for j in range(3)]);ib=np.array([np.searchsorted(cumulative[:,j],b[j],side='right') for j in range(3)])
        out.append(sorted_values[ia,np.arange(3)]*(b-rank)+sorted_values[ib,np.arange(3)]*(rank-a))
        # Equal floor/ceil ranks select the observed quantile directly.
        same=a==b
        if same.any():out[-1][same]=sorted_values[ia[same],np.flatnonzero(same)]
    bounds=np.quantile(out,[.025,.975],axis=0).T
    return dict(physical_source_sampling_units=n,test_measurements=len(vectors),median_source_resampling95=bounds[:2],radial_P68_source_resampling95=bounds[2],
        scope='Resample physical training-source identities; every repeated held-year measurement follows that source. Conditional source-sampling interval after the1arcsec/margin gate, not independent-exposure/sky confidence or stellar purity. Shared zero-point terms remain in formal prediction covariance, not assumed to average as independent sources.')


def run(root):
    selection=read(root/'registration/private_year_core_selection.json');parts=[]
    for e in selection['selected']:
        folder=root/'year'/e['core'];original=folder/'v22_year_core_summary.json';summary=read(original)
        if summary['status']!='completed':parts.append(summary);continue
        entities=read(folder/'private_training_before_test.json.gz')['entities'];records=read(folder/'private_held_test_matches.json.gz');groups=defaultdict(list)
        for r in records:groups[(r['method'],entities[r['entity']]['kind'])].append(r)
        for method,info in summary['methods'].items():
            for kind,description in info['by_class'].items():
                position=description['position'];position.pop('median_object_interval95',None)
                group=groups[(method,kind)]
                if group:position['source_sampling']=source_interval([r['native_residual_mas'] for r in group],[r['entity'] for r in group])
        summary['original_summary_sha256']=digest(original);summary['interval_readout']='Original generic per-measurement median_object_interval95 label removed in this independent export; actual whole-source resampling intervals substituted. Raw receipt unchanged.'
        parts.append(summary)
    result=dict(cores=parts,implementation_sha256=digest(Path(__file__)),slurm_job=os.environ['SLURM_JOB_ID'],scope='Frozen metadata selection and all original source/test identities; read-only uncertainty relabel and source-level sampling, no new fitted model or favourable source selection.')
    check_aggregate(result);save_json(root/'year/v22_far_core_year_validation_summary.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use Slurm')
    run(a.root)
