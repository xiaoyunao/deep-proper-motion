"""Object-held-out, physical-angle validation of a complete reference component."""
import argparse
from collections import Counter
import os
from pathlib import Path
import resource
from time import monotonic

import numpy as np
import pandas as pd

from .astrometry import project,unproject
from .blind_catalogue_quality import read
from .continuous_reference import read_offsets,reference_frame
from .coordinate_metric import forward_jacobian
from .full_reference_graph import connect
from .full_reference_run import local_support
from .io import digest,save_json
from .public_summary import check_aggregate
from .usability_cluster import summarize


def validate(root,number,output):
    start=monotonic();output.mkdir(parents=True,exist_ok=True)
    graph=root/'reference'/f'component-{number:04d}'/'private_c0_unfiltered.npz'
    center=reference_frame(graph);solution=read_offsets(graph);names=np.array(sorted(solution));draw=np.stack([solution[n][1] for n in names])
    plan=read(root/'registration/private_full_coordinate_plan.json')
    original=[Path(k) for k,v in plan['graphs'].items() if np.allclose(v['center'],center,atol=1e-12,rtol=0)]
    if len(original)!=1:raise ValueError('Old reference chart comparison needs explicit coordinate re-expression')
    old=read_offsets(original[0]);old_names=set(old)
    db=connect(root/'graph','12GB',4)
    # Hash only distinct physical identities, not every repeated measurement.
    # 1/64 is an intentional predeclared engineering/science validation sample,
    # not a silently truncated catalogue or query service maximum.
    db.execute('''CREATE OR REPLACE TEMP TABLE fixed_held AS
      SELECT reference FROM (SELECT DISTINCT reference FROM clean WHERE is_holdout)
      WHERE substr(sha256('v22-held-readout:'||reference),1,2) IN ('00','01','02','03')''')
    frame=db.execute('''SELECT c.* FROM clean c SEMI JOIN fixed_held USING(reference)
      JOIN group_map USING("group") WHERE is_holdout AND component=? ORDER BY reference,"group"''',[number]).fetchdf()
    db.close()
    hold_reference_count=frame.reference.nunique();raw=project(frame.ra_deg,frame.dec_deg,center)
    groups=frame['group'].to_numpy();known=pd.Series(groups).isin(names).to_numpy()
    keep,se=local_support(frame.ra_deg,frame.dec_deg,center,groups,names,draw)
    corrected=raw-np.array([solution[g][0] if g in solution else np.zeros(2) for g in groups])
    prior_known=pd.Series(groups).isin(old_names).to_numpy()
    previous=raw-np.array([old[g][0] if g in old else np.zeros(2) for g in groups])
    columns={};exits=Counter();entries=[]
    for ref,ix in frame.groupby('reference',sort=False).indices.items():
        ix=np.asarray(ix,int);sdss=frame.exposure_id.iloc[ix].str.startswith('SDSS:').to_numpy()
        pivot=unproject(np.median(raw[ix],axis=0),center)[0];b=np.linalg.inv(forward_jacobian(*pivot,center))
        out=dict(reference=str(ref));available=False
        for tag,values,mask in [('raw_new_support',raw,keep),('new_all_supported',corrected,keep),
            ('old_strict_common',previous,keep&prior_known),('new_strict_common',corrected,keep&prior_known),
            ('old_archived_support',previous,prior_known)]:
            early=ix[sdss&mask[ix]];late=ix[~sdss&mask[ix]]
            if not len(early) or not len(late):exits[tag+'_missing_cross_survey']+=1;continue
            delta=b@(np.median(values[early],axis=0)-np.median(values[late],axis=0))
            columns.setdefault(tag,[]).append(delta);out[tag]=delta;available=True
        if available:entries.append(out)
    # Paired ratios use exactly the same retained held sources/detections.
    common=[e for e in entries if {'old_strict_common','new_strict_common'}<=e.keys()]
    old_values=np.array([e['old_strict_common'] for e in common]);new_values=np.array([e['new_strict_common'] for e in common])
    result=dict(component=number,reference_model_sha256=digest(graph),old_reference_model_sha256=digest(original[0]),
        validation_physical_sources=hold_reference_count,validation_measurements=len(frame),
        native12mas_supported_measurements=int(keep.sum()),old_archived_group_measurements=int(prior_known.sum()),
        sample='Physical holdout identity sha256(v22-held-readout:ID) first byte<04,1/64; reference training/holdout split unchanged',
        position_mas={k:summarize(v) for k,v in columns.items()},exits=dict(exits),
        strict_same_physical_detections_and_objects=dict(n=len(common),
            radial_p68_new_over_old=float(np.quantile(np.linalg.norm(new_values,axis=1),.68)/np.quantile(np.linalg.norm(old_values,axis=1),.68)) if len(common)>=5 else None),
        elapsed_seconds=monotonic()-start,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        slurm_job=os.environ['SLURM_JOB_ID'],
        scope='Complete new graph, reference-source heldout sample, native E/N cross-survey median position difference. Strict old/new entries use identical detections (new native support AND old group presence). New-all and old-archived supports separate; no Gaia or QSO fitted. Object intervals condition on shared physical observations; not independent sky confidence.')
    check_aggregate(result);save_json(output/'v22_complete_reference_validation_summary.json',result)
    save_json(output/'private_held_reference_validation.json',entries)
    print('FULL_REFERENCE_HELD_VALIDATED',number,len(common),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('component',type=int);p.add_argument('output',type=Path);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use Slurm')
    validate(**vars(a))
