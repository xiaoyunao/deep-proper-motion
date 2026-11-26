"""Freeze the original102 bright optical/field pairs and inspect actual groups."""
import argparse
from collections import defaultdict
from functools import lru_cache
import os
from pathlib import Path

import numpy as np

from .blind_catalogue_quality import read
from .continuous_reference import read_offsets
from .coordinate_metric import forward_jacobian
from .dcr_geometry import sdss_colors,combine_color
from .io import save_json,digest
from .public_summary import check_aggregate
from .stellar_g_mechanism_v22 import COEFFICIENTS,deep_neighbours,finite_json
from .usability_cluster import summarize


def run(root,lsroot,output,sas=None):
    output.mkdir(parents=True,exist_ok=True);input=root/'v2.1/stellar-attribution/private_stellar_positions.json.gz';p=read(input)
    plan=read(root/'v2.1/registration/private_coordinate_plan.json');lookup={r['core']:r for r in plan['selected']}
    perstar=defaultdict(list)
    for m in p['measurements']:perstar[tuple(m['source_key'])].append(m)
    chosen=[r for r in p['states'] if 'c0_residual' in r and r.get('color') is not None and np.isfinite(r['rmag'])]
    target=[r for r in chosen if r['old_target']];field=[r for r in chosen if not r['old_target']]
    def feature(r):
        m=perstar[tuple(r['source_key'])]
        return (frozenset(v['calibration_group'] for v in m if v['survey']=='SDSS'),frozenset(v['band'] for v in m if v['survey']=='NSC'),
            float(np.ptp([v['time'] for v in m])),float(m[0]['crowding_mas']))
    features={tuple(r['source_key']):feature(r) for r in chosen};used=set();pairs=[]
    for r in sorted(target,key=lambda r:r['source_key']):
        a=features[tuple(r['source_key'])];candidates=[]
        for j,s in enumerate(field):
            if j in used or abs(r['rmag']-s['rmag'])>.3 or abs(r['color']-s['color'])>.15:continue
            b=features[tuple(s['source_key'])]
            if a[0]!=b[0] or len(a[1]&b[1])/max(1,len(a[1]|b[1]))<.8 or abs(a[2]-b[2])>2:continue
            if min(a[3],b[3])<2000 or abs(np.log10(a[3]/b[3]))>.3:continue
            candidates.append((abs(r['rmag']-s['rmag'])/.3+abs(r['color']-s['color'])/.15,j))
        if not candidates:continue
        j=min(candidates)[1];used.add(j);s=field[j];pairs.append((r,s))
    if len(pairs)!=102:raise ValueError('Frozen102 metadata pairing did not reproduce')
    keys=sorted({tuple(r['source_key']) for pair in pairs for r in pair})
    old_entries=read(root/'v2.0/ngc_motion/private_cluster_diagnostics.json.gz')['entries']
    physical_ids={tuple(r['source_key']):r['row']['detection_ids'] for r in old_entries}
    field_sets={};colour={};photometry_receipts={}
    all_sdss=set()
    for key in keys:
        ids=[str(v) for v in physical_ids[key] if str(v).startswith('SDSS:')]
        all_sdss.update(ids);parsed=[v for v in ids if len(v.split(':'))==5]
        field_sets[key]=set(':'.join(v.split(':')[:4]) for v in parsed) if len(parsed)==len(ids) else None
    if sas is not None:
        phot,photometry_receipts=sdss_colors(sorted(all_sdss),sas);phot=phot.set_index('detection_id')
        for key in keys:
            ids=[v for v in physical_ids[key] if v in phot.index]
            colour[key]=combine_color(phot.reindex(ids))

    sky=np.array([[perstar[k][0]['ra_deg'],perstar[k][0]['dec_deg']] for k in keys]);deep,receipts=deep_neighbours(sky,lsroot);neighbours=dict(zip(keys,deep))
    g=read(root/'v2.2/stellar-g/private_stellar_g_mechanism.json')['measurements'];gpred=defaultdict(lambda:np.zeros(2));gvector=defaultdict(lambda:np.zeros(2))
    for m in g:
        key=tuple(m['source_key'])
        if key not in neighbours:continue
        entry=lookup[key[0]];a=forward_jacobian(m['ra_deg'],m['dec_deg'],entry['graph_center'])
        v=np.asarray(m['k_icrs'])@a@np.asarray(m['native_geometry']);gvector[key]+=v;gpred[key]+=v*(m['color']-1.2)
    @lru_cache(maxsize=4)
    def offsets(path):return read_offsets(Path(path))
    response={}
    for key in keys:
        entry=lookup[key[0]];zero=offsets(entry['graph']);draw=np.zeros((48,2))
        for m in perstar[key]:draw-=np.asarray(zero[m['calibration_group']][1])@np.asarray(m['k_icrs']).T
        response[key]=draw
    results=[]
    def jac(a,b):return len(a&b)/max(1,len(a|b))
    for r,s in pairs:
        a,b=tuple(r['source_key']),tuple(s['source_key']);m,n=perstar[a],perstar[b]
        def selected(v,survey):return {q['physical_group'] for q in v if q['survey']==survey}
        def exp(v):return {q['calibration_group'] for q in v if q['survey']=='NSC'}
        residual=np.asarray(r['c0_residual'])-r['parallax_slope']-np.asarray(s['c0_residual'])+s['parallax_slope']
        # Each graph retains its own synchronized draw labels. For the NGC
        # sample these must be one shared graph before median propagation.
        if lookup[a[0]]['graph']!=lookup[b[0]]['graph']:raise ValueError('New graph pairing needs independent graph MonteCarlo, not aligned draw labels')
        results.append(dict(source_keys=[a,b],residual=residual,g_unit_coefficient_prediction=gpred[a]-gpred[b],
            shared_reference_draws=response[a]-response[b],graph_sha256=lookup[a[0]]['graph_sha256'],
            same_SDSS_fields=(field_sets[a]==field_sets[b]) if field_sets[a] is not None and field_sets[b] is not None else None,
            SDSS_field_jaccard=jac(field_sets[a],field_sets[b]) if field_sets[a] is not None and field_sets[b] is not None else None,
            NSC_exposure_jaccard=jac(exp(m),exp(n)),NSC_CCD_band_jaccard=jac(selected(m,'NSC'),selected(n,'NSC')),
            deep_neighbour=[neighbours[a],neighbours[b]],
            own_C0_difference=np.asarray(r['c0_predicted_change'])-s['c0_predicted_change']))
    by={}
    for name,fn in [('all',lambda r:True),('same_SDSS_fields',lambda r:r['same_SDSS_fields'] is True),('different_SDSS_fields',lambda r:r['same_SDSS_fields'] is False),
        ('NSC_exposure_jaccard_ge0p8',lambda r:r['NSC_exposure_jaccard']>=.8),('NSC_exposure_jaccard_lt0p8',lambda r:r['NSC_exposure_jaccard']<.8),
        ('NSC_CCD_jaccard_ge0p8',lambda r:r['NSC_CCD_band_jaccard']>=.8),('deep_both3arcsec',lambda r:all(v.get('neighbour_distance_mas',0)>=3000 for v in r['deep_neighbour']))]:
        part=[r for r in results if fn(r)];out=dict(n=len(part))
        if len(part)>=5:
            out.update(residual=summarize([r['residual'] for r in part]),own_C0_difference=summarize([r['own_C0_difference'] for r in part]),
                g_frozen_predictions={str(k):summarize([k*np.asarray(r['g_unit_coefficient_prediction']) for r in part]) for k in COEFFICIENTS},
                after_g_only_prediction={str(k):summarize([r['residual']-k*np.asarray(r['g_unit_coefficient_prediction']) for r in part]) for k in COEFFICIENTS})
        by[name]=out
    data=np.asarray([r['residual'] for r in results]);draw=np.asarray([r['shared_reference_draws'] for r in results])
    graph_count=len({r['graph_sha256'] for r in results})
    if graph_count!=1:raise ValueError('Median shared uncertainty must use independent graph draws')
    median_draw=np.median(data[:,None,:]+draw,axis=0)-np.median(data,axis=0)
    mean_draw=draw.mean(axis=0)
    save_json(output/'private_ngc102_group_attribution.json',finite_json(dict(pairs=results,LS_file_receipts=receipts,photo_file_receipts=photometry_receipts,
        source_colours=[dict(source_key=k,published_color=colour.get(k),full_trajectory_unit_g_response=gvector[k]) for k in keys])))
    report=dict(input_sha256=digest(input),frozen_metadata_pairs=len(pairs),strata=by,
        synchronized_reference_median_shift_se_masyr=np.std(median_draw,axis=0,ddof=1),
        synchronized_reference_equal_mean_shift_se_masyr=np.std(mean_draw,axis=0,ddof=1),
        physical_SDSS_mapping=dict(physical_records=len(all_sdss),five_token_SAS_mapped=sum(len(v.split(':'))==5 for v in all_sdss),physical_fields=len(photometry_receipts),SAS_records_recovered=len(phot) if sas is not None else 0,
            sources_with_colour_error=sum(np.isfinite(c+e) and e>0 for c,e in colour.values()),
            median_gi_error=float(np.nanmedian([e for c,e in colour.values() if np.isfinite(c+e) and e>0])) if any(np.isfinite(c+e) and e>0 for c,e in colour.values()) else None,
            scope='Physical run/camcol/field/ID -> published PSFMAG/PSFMAGERR, g-i error independent-band approximation; no new position. Target colour error is shared over its full g response, not per-detection independent. Not a joint adopted colour/coefficient/C0 covariance model.'),
        graph_count=graph_count,slurm_job=os.environ['SLURM_JOB_ID'],
        scope='Exactly reproduce102 frozen bright optical/field pairs, no PM reselection. Gaia PM and known annual-parallax slope removed only for validation. Actual SDSS field and NSC exposure/CCD sets now compared. K from saved full2D source estimator; old C0 graph draws propagated with that K, then mean/median across pairs synchronously, not divided as independent sources. g-only frozen-k prediction is a diagnostic; colour error and coefficient/group joint model not certified. No new production correction.')
    check_aggregate(report);save_json(output/'v22_ngc102_east_attribution_summary.json',report)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('root','lsroot','output'):p.add_argument(k,type=Path)
    p.add_argument('--sas',type=Path)
    a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use Slurm')
    run(**vars(a))
