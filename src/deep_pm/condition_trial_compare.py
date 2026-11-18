"""Compare saved multiband fits with releases on strictly common QSO detections."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.time import Time

from .continuous_reference import reference_frame,verify_directory
from .continuous_summary import estimate
from .io import digest,save_json
from .nsc_comparison import comparison
from .public_summary import check_aggregate
from .astrometry import project
from .standard_motion import icrs_motion


def join_controls(trial,baseline):
    lookup={frozenset(r['detection_ids']):r for r in baseline}
    if len(lookup)!=len(baseline):
        raise ValueError('Baseline QSO detection identities are not unique')
    joined=[]
    for row in trial:
        key=frozenset(row['r_detection_ids'])
        if key not in lookup:
            raise ValueError('Trial changed the fixed r detection identities')
        old=lookup[key]
        if old['magnitude']!=row['magnitude'] or not np.allclose(row['values']['W0_r'][0],old['motion']['all_SDSS'],rtol=0,atol=1e-7):
            raise ValueError('Trial W0-r baseline failed reproduction')
        joined.append((row,old))
    return joined


def run(trial,conditions,source,output):
    verify_directory(trial,('private_error_pairs.json','error_trial_summary.json'))
    verify_directory(conditions,('private_conditions.json','conditions_summary.json'))
    if digest(trial/'source_receipt.json')!=digest(conditions/'source_receipt.json'):
        raise ValueError('Source input receipts differ')
    joined=join_controls(json.loads((trial/'private_error_pairs.json').read_text()),
                         json.loads((conditions/'private_conditions.json').read_text()))
    standard=all('standard_values' in new for new,old in joined)
    center=reference_frame((trial/'shared_receipt.json').resolve().parent/'private_c0.npz') if standard else None
    table=pd.read_csv(source/'detections.csv',usecols=['detection_id','exposure_id','observation_time']).set_index('detection_id')
    records=[]
    for new,old in joined:
        if old['published'] is None or old['legacy'] is None:
            continue
        rows=table.loc[new['multi_detection_ids']]
        nsc=rows.exposure_id.str.startswith('NSC:')
        times=Time(rows.observation_time.to_numpy(),format='mjd',scale='tt').utc.mjd
        values=dict(new['values'],NSC=old['published'],Xiao=old['legacy'])
        if standard:
            position=project([new['position_deg'][0]],[new['position_deg'][1]],center)[0]
            jac=icrs_motion(position,np.zeros(2),center)['state_jacobian'][np.ix_([1,3],[1,3])]
            values=dict(new['standard_values'],**{key:(jac@np.asarray(values[key][0]),values[key][1]) for key in ('NSC','Xiao')})
        records.append(dict(magnitude=new['magnitude'],values=values,
                            exposures=sorted(set(rows.exposure_id)),nsc_epoch_days=sorted(set(np.floor(times[nsc]-.75).astype(int)))))
    settings=dict(min_publish_count=3,tail_masyr=5,bootstrap_draws=800,bootstrap_absolute_zero=True,seed=9137)
    results={}
    for low in (20,21,22):
        part=[r for r in records if low<=r['magnitude']<low+1]
        contrasts=(('W0_multi','Xiao'),('W0_multi','NSC'),('W0_multi','W0_r'),('W2_r','Xiao'),('W2_multi','Xiao'))
        stats=comparison([r['values'] for r in part],settings,contrasts)
        weighted={}
        if len(part)>=5:
            rows=[dict(r,motion={k:v[0] for k,v in r['values'].items()}) for r in part]
            methods=tuple(part[0]['values'])
            for candidate in ('W0_r','W0_multi','W2_multi'):
                weighted[candidate]=estimate(rows,methods,9137,candidate=candidate)
        results[f'r{low}_{low+1}']=dict(common=stats,physical_group_sensitivity=weighted)
    public=dict(fixed_r_reproduction_count=len(joined),common_count=len(records),cohorts=results,
        motion_frame='ICRS with full fitted-state covariance; published native marginal errors' if standard else 'original tangent frame; published native marginal errors',
        input_summary_sha256=digest(trial/'error_trial_summary.json'),baseline_summary_sha256=digest(conditions/'conditions_summary.json'),
        physical_weight_scope='Union of actual r and multiband observations; NSC observing nights grouped at 18:00 UTC. Source-weight sensitivity, not independent spatial confidence.',
        slurm_job=os.environ['SLURM_JOB_ID'])
    check_aggregate(public)
    save_json(output,public)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('trial','conditions','source','output'):
        p.add_argument(name,type=Path)
    a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        p.error('Use a Slurm compute node')
    run(a.trial,a.conditions,a.source,a.output)


if __name__=='__main__':
    main()
