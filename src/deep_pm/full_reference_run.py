"""Solve a complete registered reference component and publish its native support.

All groups are retained in the unfiltered graph. The native12mas application
selection is per physical target position, not a one-number graph-plane cut.
"""
import argparse
import json
import os
from pathlib import Path
import resource
from time import monotonic

import numpy as np
import pandas as pd

from .blind_catalogue_quality import read
from .coordinate_metric import forward_jacobian
from .full_reference_solver import fit_component
from .io import digest,save_json
from .public_summary import check_aggregate


def local_support(ra,dec,center,groups,names,draws,limit=12.):
    """Group covariance in graph units -> each measurement's local ICRS basis."""
    index=pd.Index(names).get_indexer(groups)
    valid=index>=0
    se=np.full((len(index),2),np.inf)
    if valid.any():
        samples=np.asarray(draws,float)
        centered=samples-samples.mean(axis=1,keepdims=True)
        covariance=np.einsum('gdi,gdj->gij',centered,centered)/(samples.shape[1]-1)
        b=np.linalg.inv(forward_jacobian(np.asarray(ra)[valid],np.asarray(dec)[valid],center))
        native=b@covariance[index[valid]]@b.transpose(0,2,1)
        se[valid]=np.sqrt(np.maximum(native[:,[0,1],[0,1]],0))
    return valid&np.isfinite(se).all(axis=1)&(se.max(axis=1)<=limit),se


def solve(root,number,draw_workers=1):
    start=monotonic();dest=root/'reference'/f'component-{number:04d}'
    binding_path=dest/'private_numeric_binding.json';binding=read(binding_path)
    for filename,sha in binding['files'].items():
        if digest(dest/filename)!=sha:raise ValueError('Changed numeric component input')
    ticket=dict(numeric_sha256=digest(binding_path),implementation_sha256=digest(Path(__file__)),
        solver_sha256=digest(Path(__file__).with_name('full_reference_solver.py')),bootstrap_draws=48,seed=45116+number,
        final_IRLS='6updates then finalsolve',bootstrap_IRLS='frozen final',native_application_se_mas=12)
    ticket_path=dest/'private_solution_binding.json'
    if ticket_path.exists() and read(ticket_path)!=ticket:raise ValueError('Changed solver version; create independent component output')
    save_json(ticket_path,ticket)
    ptr,groups,precision,values,names=[np.load(dest/(key+'.npy'),mmap_mode='r') for key in ('ptr','groups','precision','values','names')]
    point,draws,weight,trace=fit_component(ptr,groups,precision,values,names,draws=48,seed=45116+number,checkpoint=dest/'checkpoints',draw_workers=draw_workers)
    counts=np.bincount(groups,minlength=len(names))
    center=binding['component']['center_deg']
    # Exact distinct source counts exported with the globally peeled graph.
    support=np.load(dest/'reference_counts.npy')
    if len(support)!=len(names) or support.min()<10:raise ValueError('Reference support fell below registered minimum')
    data=dict(names=names,position=point,draws=draws,counts=support,
        gauge=np.repeat(str(number),len(names)),projection_center_deg=np.asarray(center),
        gauge_nsc_names=names[np.char.startswith(names,'NSC:')],native_application_se_mas=np.array(12.),
        unfiltered=np.array(True),bootstrap_IRLS=np.array('frozen_final'))
    if "projection_kind" in binding["component"]:
        data["projection_kind"]=np.array(binding["component"]["projection_kind"])
    np.savez_compressed(dest/'private_c0_unfiltered.npz',**data)
    # The conventional basename also contains ALL solved groups. New callers
    # must apply local_support; it is deliberately not an old planar-SE graph.
    np.savez_compressed(dest/'private_c0.npz',**data)
    gauge_sha=digest_text('\n'.join(data['gauge_nsc_names']))
    summary=dict(component=number,center_deg=center,groups=len(names),measurements=len(groups),reference_sources=len(ptr)-1,
        gauge_nsc_group_count=len(data['gauge_nsc_names']),gauge_set_sha256=gauge_sha,
        graph_sha256=digest(dest/'private_c0_unfiltered.npz'),numeric_binding_sha256=digest(binding_path),
        graph_plane_se_p16_p50_p84=np.quantile(np.std(draws,axis=1,ddof=1),[.16,.5,.84],axis=0),
        group_measurement_p16_p50_p84=np.quantile(counts,[.16,.5,.84]),minimum_reference_count=int(support.min()),
        convergence_history=trace,elapsed_seconds=monotonic()-start,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        slurm_job=os.environ['SLURM_JOB_ID'],draw_workers=draw_workers,scope='Complete physical connected component; float64 full2D Schur; native50mas before projection; final point/weight consistent; 48 synchronous source bootstrap; pre-SE NSC median gauge; unfiltered groups, per-target localICRS12mas support downstream.')
    if "projection_kind" in binding["component"]:
        summary.update(projection_kind=binding["component"]["projection_kind"],
                       maximum_linear_scale=binding["component"]["maximum_linear_scale"])
    check_aggregate(summary);save_json(dest/'v22_reference_component_summary.json',summary)
    print('REFERENCE_COMPONENT_DONE',number,len(names),flush=True)


def digest_text(value):
    from hashlib import sha256
    return sha256(value.encode()).hexdigest()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('component',type=int);p.add_argument('--draw-workers',type=int,default=1);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use Slurm')
    if not 1<=a.draw_workers<=int(os.environ.get("SLURM_CPUS_PER_TASK",1)):p.error("Use allocated CPUs")
    solve(a.root,a.component,a.draw_workers)
