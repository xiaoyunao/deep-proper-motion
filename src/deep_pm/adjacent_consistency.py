"""Same-source boundary checks with shared detections and C0 draws retained."""
import argparse
import gzip
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .astrometry import project
from .continuous_reference import verify_directory
from .dark_precision import query_unique
from .exposure_catalogues import coordinates
from .io import digest,save_json
from .public_summary import check_aggregate
from .standard_motion import icrs_motion


def common_frame(row,center):
    position=project([row['position_deg'][0]],[row['position_deg'][1]],center)[0]
    j=icrs_motion(position,np.zeros(2),center)['state_jacobian'][np.ix_([1,3],[1,3])]
    velocity=np.linalg.solve(j,np.asarray(row['pm_icrs_masyr']))
    inverse=np.linalg.inv(icrs_motion(position,velocity,center)['state_jacobian'])
    transform=inverse@np.asarray(row['state_jacobian'])
    return np.array([position[0],velocity[0],position[1],velocity[1]]),transform


def difference_covariance(first,second):
    if not np.array_equal(first['projection_center_deg'],second['projection_center_deg']):
        raise ValueError('Shared-measurement covariance needs a common physical measurement frame')
    center=tuple(first['position_deg'])
    a,ta=common_frame(first,center)
    b,tb=common_frame(second,center)
    effects={}
    for sign,row,transform in ((-1,first,ta),(1,second,tb)):
        h=np.einsum('pq,qna->pna',transform,np.asarray(row['response']))
        for i,key in enumerate(row['detection_ids']):
            sigma=np.asarray(row['measurement_sigma_mas'][i])
            if key in effects:
                old_s,old_h=effects[key]
                if not np.allclose(old_s,sigma,rtol=1e-10,atol=1e-10):
                    raise ValueError('One physical detection has inconsistent measurement errors')
                effects[key]=sigma,old_h+sign*h[:,i,:]
            else:
                effects[key]=sigma,sign*h[:,i,:]
    independent=sum((h*sigma**2)@h.T for sigma,h in effects.values())
    ra=np.asarray(first['reference_state_draws'])@ta.T
    rb=np.asarray(second['reference_state_draws'])@tb.T
    if ra.shape!=rb.shape:
        raise ValueError('Reference draws must be synchronized')
    common=np.cov(rb-ra,rowvar=False)
    return b-a,independent,common


def audit(left,right,output):
    names=('continuous_core_summary.json','private_standard_motion.json.gz')
    verify_directory(left,names);verify_directory(right,names)
    summaries=[json.loads((p/'continuous_core_summary.json').read_text()) for p in (left,right)]
    if summaries[0]['shared_solution_sha256']!=summaries[1]['shared_solution_sha256']:
        raise ValueError('Boundary comparison needs a shared physical reference solution')
    records=[]
    for p in (left,right):
        with gzip.open(p/'private_standard_motion.json.gz','rt') as f:
            records.append(json.load(f))
    a,b=records
    pairs=[]
    if a and b:
        target=np.asarray([r['position_deg'] for r in a])
        other=np.asarray([r['position_deg'] for r in b])
        radius=2*np.sin(np.deg2rad(.2/3600)/2)
        margin=2*np.sin(np.deg2rad(.1/3600)/2)
        good,idx=query_unique(cKDTree(coordinates(*other.T)),coordinates(*target.T),radius,margin)
        for i in np.flatnonzero(good):
            delta,ind,ref=difference_covariance(a[i],b[idx[i]])
            pairs.append(dict(delta=delta,independent=ind,reference=ref,
                shared_detections=len(set(a[i]['detection_ids'])&set(b[idx[i]]['detection_ids'])),
                shared_groups=len(set(a[i]['physical_groups'])&set(b[idx[i]]['physical_groups']))))
    result=dict(left_count=len(a),right_count=len(b),unique_common_count=len(pairs),
        shared_solution_sha256=summaries[0]['shared_solution_sha256'],
        source_summaries_sha256=[digest(p/'continuous_core_summary.json') for p in (left,right)],
        covariance_scope='Linear response to the same physical measurement errors and synchronous C0 draws; original IRLS weights held fixed.',
        slurm_job=os.environ['SLURM_JOB_ID'])
    if pairs:
        delta=np.array([p['delta'] for p in pairs])
        result.update(median_position_difference_mas=np.median(delta[:,[0,2]],axis=0),
            median_motion_difference_masyr=np.median(delta[:,[1,3]],axis=0),
            radial_position_p68_mas=np.quantile(np.linalg.norm(delta[:,[0,2]],axis=1),.68),
            radial_motion_p68_masyr=np.quantile(np.linalg.norm(delta[:,[1,3]],axis=1),.68),
            shared_detection_count_quantiles=np.quantile([p['shared_detections'] for p in pairs],[.16,.5,.84]),
            shared_group_count_quantiles=np.quantile([p['shared_groups'] for p in pairs],[.16,.5,.84]),
            linearized_variance_quantiles=np.quantile([np.diag(p['independent']+p['reference']) for p in pairs],[.16,.5,.84],axis=0))
    check_aggregate(result)
    save_json(output,result)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('left','right','output'):
        p.add_argument(name,type=Path)
    a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        p.error('Use a Slurm compute node')
    audit(a.left,a.right,a.output)


if __name__=='__main__':
    main()
