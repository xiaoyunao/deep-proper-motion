"""Minimal held-galaxy error learning with independent QSO and year tests."""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .__main__ import tomllib
from .astrometry import project, unproject
from .astrometry_weights import variance_model
from .continuous_reference import FILES, PRECISION, SPATIAL, bind, read_offsets, reference_frame, verify_directory
from .condition_holdout import fixed_year_prediction
from .exposure_catalogues import inside
from .io import save_json
from .nsc_comparison import qso_matches, comparison
from .pilot_production import quality_detections, valid_epochs
from .precision_v03 import calibration_groups, fit_with_calibration
from .provenance import run_stage
from .public_summary import check_aggregate
from .regional_production import prediction


def split_reference(name, seed):
    return int(sha256(f'{seed}:error:{name}'.encode()).hexdigest()[:8],16)/2**32 < .5


def disjoint_pairs(index, time, groups):
    """No repeated detection, group, or night within a reference's paired arm."""
    index = sorted(index,key=lambda i: time[i])
    available, used_groups, used_nights = [], set(), set()
    for i in index:
        night = int(np.floor(time[i]*365.25))
        if groups[i] not in used_groups and night not in used_nights:
            available.append(i)
            used_groups.add(groups[i]); used_nights.add(night)
    return [(available[k],available[k+1]) for k in range(0,len(available)-1,2)]


def families(data):
    instrument = data.get('instrument',pd.Series('unknown',index=data.index)).fillna('unknown').astype(str)
    label = 'NSC:'+instrument+':'+data.band.astype(str)
    label.loc[data.survey.eq('SDSS')] = 'SDSS:r'
    return label.to_numpy()


def learn_errors(refs, data, coord, time, sigma, keys, seed, minimum, reference_offsets=None):
    lookup = pd.Series(data.index,index=data.detection_id)
    selected = refs.loc[refs.is_holdout].copy()
    selected['input_index'] = selected.detection_id.map(lookup)
    selected = selected.loc[selected.input_index.notna()]
    selected['input_index'] = selected.input_index.astype(int)
    selected['error_training'] = selected.reference.map(lambda k: split_reference(k,seed))
    family = families(data)
    selected['family'] = family[selected.input_index]
    pairs = defaultdict(list)
    counts = Counter()
    for (ref,label), part in selected.loc[selected.error_training].groupby(['reference','family'],sort=False):
        indices = part.input_index.to_numpy()
        for a,b in disjoint_pairs(indices,time,keys):
            pairs[label].append((coord[a]-coord[b],sigma[a],sigma[b]))
        counts[label] += 1
    raw = data[['sigma_east_mas','sigma_north_mas']].to_numpy(float)
    revised = np.hypot(raw,50.)
    models = {}
    for label, values in pairs.items():
        delta,_,_ = map(np.asarray,zip(*values))
        # Fit the raw marginal errors, rather than multiplying an already-added floor.
        part = selected.loc[selected.error_training & selected.family.eq(label)]
        raw_pairs = []
        for _, group in part.groupby('reference',sort=False):
            raw_pairs.extend(disjoint_pairs(group.input_index.to_numpy(),time,keys))
        left,right = np.array(raw_pairs,dtype=int).T
        common = np.array([np.var(reference_offsets[keys[a]][1]-reference_offsets[keys[b]][1],axis=0,ddof=1)
                           for a,b in raw_pairs]) if reference_offsets else np.zeros((len(raw_pairs),2))
        parameters,valid = [],[]
        for axis in range(2):
            parameters_axis, supported = variance_model(delta[:,axis],raw[left,axis],raw[right,axis],minimum,
                                                        common_variance=common[:,axis])
            parameters.append(parameters_axis); valid.append(supported)
        parameters = np.array(parameters)
        if all(valid):
            use = family == label
            revised[use] = np.sqrt((raw[use]*parameters[:,0])**2+parameters[:,1]**2)
        models[label] = dict(pair_count=len(values),reference_count=counts[label],parameters_scale_floor=parameters,
                             fitted_by_axis=valid,enabled_for_trial_only=bool(all(valid)),
                             reference_pair_variance_median_mas2=np.median(common,axis=0))
    return revised,models,selected


def run(work, source, reference, shared, spec, protocol):
    config = tomllib.loads(PRECISION.read_text())
    settings = tomllib.loads(SPATIAL.read_text())
    settings.update(bootstrap_draws=400,bootstrap_absolute_zero=True)
    center = reference_frame(shared/'private_c0.npz')
    data,xy,time,sigma,keep = quality_detections(source,center,config,settings['production'])
    keys = calibration_groups(data)
    offsets,gauges = read_offsets(shared/'private_c0.npz',with_gauge=True)
    on = inside(data.ra_deg,data.dec_deg,spec)&keep
    counts = pd.Series(keys[on]).map(gauges).value_counts()
    gauge = min(counts.index,key=lambda k:(-counts[k],k)) if len(counts) else None
    offsets = {k:v for k,v in offsets.items() if gauges[k]==gauge}
    support = np.isin(keys,list(offsets))
    coord = xy-np.stack([offsets[k][0] if k in offsets else np.zeros(2) for k in keys])
    refs = pd.read_csv(reference/'reference.csv.gz',dtype={'reference':str})
    refs = refs.loc[refs.group.isin(offsets)]
    revised,models,held = learn_errors(refs,data,coord,time,sigma,keys,protocol['seed'],protocol['min_variance_pairs'],offsets)
    qso = pd.read_csv(source/'spectra_union.csv')
    qso = qso.loc[inside(qso.ra,qso.dec,spec)].reset_index(drop=True)
    nsc = data.survey.eq('NSC').to_numpy()
    expanded = keep | (nsc & data.band.isin(['g','i','z']).to_numpy()
                       & data.magnitude_error.between(0,2.5/np.log(10)/5,inclusive='right').to_numpy())
    _,truth_r = qso_matches(data,xy,keep,project(qso.ra,qso.dec,center))
    _,truth_multi = qso_matches(data,xy,expanded,project(qso.ra,qso.dec,center))
    records,exits = [],Counter()
    for k,index in truth_r.items():
        if not 20 <= qso.reference_r.iloc[k] < 23:
            continue
        ix = index[support[index]]
        if not valid_epochs(ix,nsc,time,settings['production']):
            exits['time_or_support'] += 1
            continue
        multi = truth_multi.get(k,np.empty(0,int))
        multi = multi[support[multi]]
        if not set(ix).issubset(multi):
            raise ValueError('Multiband matching changed the r detection identities')
        values = {}
        for name,use,error in [('W0_r',ix,sigma),('W2_r',ix,revised),('W0_multi',multi,sigma),('W2_multi',multi,revised)]:
            fit,err = fit_with_calibration(xy[use],time[use],error[use],config['reference_epoch'],
                                            [offsets[keys[j]] for j in use])
            values[name] = (fit.coeff[1],err)
        records.append(dict(magnitude=qso.reference_r.iloc[k],values=values,
                            r_detection_ids=data.detection_id.iloc[ix].tolist(),
                            multi_detection_ids=data.detection_id.iloc[multi].tolist()))
    contrasts = (('W2_r','W0_r'),('W0_multi','W0_r'),('W2_multi','W0_r'),('W2_multi','W2_r'))
    qso_summary = {f'r{low}_{low+1}': comparison([r['values'] for r in records if low <= r['magnitude'] < low+1],settings,contrasts)
                   for low in (20,21,22)}
    galaxy = {}
    for sample,mask in [('training',held.error_training),('validation',~held.error_training)]:
        pairs = defaultdict(list)
        for (_,label), part in held.loc[mask].groupby(['reference','family'],sort=False):
            for a,b in disjoint_pairs(part.input_index.to_numpy(),time,keys):
                delta = coord[a]-coord[b]
                pairs[label].append([delta,delta/np.hypot(sigma[a],sigma[b]),delta/np.hypot(revised[a],revised[b])])
        galaxy[sample] = {label:dict(pair_count=len(v),median_and_standardized_median=np.median(v,axis=0),
            nmad_and_standardized_nmad=1.4826*np.median(abs(np.asarray(v)-np.median(v,axis=0)),axis=0))
            for label,v in pairs.items()}
    def core(pos):
        return inside(*unproject(pos,center)[0],spec)
    # Both held-year arms use the SAME trial errors; isolates adding bands.
    annual = prediction(work,center,data,xy,time,revised,revised,keys,nsc,keep,expanded,offsets,
                         settings,config,dict(radius=9.,core_members=core))
    fixed_annual,held_pairs = fixed_year_prediction(data,xy,time,sigma,revised,keys,nsc,keep,expanded,
                                                   offsets,settings,config,spec,center)
    save_json(work/'private_error_pairs.json',records)
    save_json(work/'private_held_error_pairs.json',held_pairs)
    summary = dict(error_models=models,independent_galaxy_pairs=galaxy,qso=qso_summary,
                   annual_same_errors_multiband=annual,fixed_source_held_year=fixed_annual,exits=exits,
                   adoption_status='research_trial_only_pending_independent_replication',
                   training_scope='Half of C0 held-out physical galaxies; QSO excluded from error learning.')
    check_aggregate(summary)
    save_json(work/'error_trial_summary.json',summary)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('work','source','reference','shared','geometry_config','protocol'):
        p.add_argument(name,type=Path)
    p.add_argument('region')
    a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        p.error('Use a Slurm compute node')
    a.work.mkdir(parents=True,exist_ok=True)
    bind(a.work,'source_receipt.json',verify_directory(a.source,FILES))
    bind(a.work,'reference_receipt.json',verify_directory(a.reference,('reference.csv.gz','reference_summary.json')))
    bind(a.work,'shared_receipt.json',verify_directory(a.shared,('private_c0.npz',)))
    bind(a.work,'geometry_config.toml',a.geometry_config)
    with run_stage(a.work,a.protocol,a.region,'conditions-error-trial-v13',
        inputs=('source_receipt.json','reference_receipt.json','shared_receipt.json','geometry_config.toml'),
        outputs=('private_error_pairs.json','private_held_error_pairs.json','error_trial_summary.json')) as go:
        if go:
            run(a.work,a.source,a.reference,a.shared,tomllib.loads(a.geometry_config.read_text())['fields'][a.region],
                tomllib.loads(a.protocol.read_text()))


if __name__=='__main__':
    main()
