"""Attach measured quality scope and verify the fixed NGC cohort after graph replay."""
import argparse
from collections import Counter,defaultdict
import gzip
import json
import os
from pathlib import Path

import healpy as hp
import numpy as np

from .blind_catalogue_quality import read
from .io import digest,save_json
from .ngc5466_demo import footprints
from .nsc_ownership_overlay import owned_records
from .public_summary import check_aggregate


def run(current,quality,cluster,output):
    output.mkdir(parents=True,exist_ok=True)
    file=current/'continuation_production_science_summary.json';wide=read(file)
    q=read(quality);support=wide['observing_support_counts'];total=wide['research_fits']
    if sum(support.values())!=total:raise ValueError('Quality classes lost rows')
    cells={};occupied=Counter()
    for row in wide['cores'].values():
        for pixel,count in row['nside128_counts'].items():
            ra,dec=hp.pix2ang(128,int(pixel),lonlat=True)
            occupied[int(hp.ang2pix(32,ra,dec,lonlat=True))]+=count
    for pixel,count in sorted(occupied.items()):
        layers={str(low):q['spatial'].get(f'{low}__{pixel}') for low in (21,22)}
        if not any(layers.values()):continue
        cells[str(pixel)]=dict(all_magnitude_source_count=count,nside=32,
            qso_quality_by_reference_r={k:dict(n=v['n'],median_masyr=v['median'],
                raw_1sigma=v['raw_1sigma'],raw_2sigma=v['raw_2sigma'],centered_1sigma=v['centered_1sigma'],
                conditional_object_zero_interval95=v['median_object_interval95'],
                independent_group_support=v['physical_support_sufficient'],qso_axis_zero_goal=v['qso_zero_goal_by_axis']) for k,v in layers.items() if v},
            stellar_transfer_certified=False,unmeasured_shared_systematic_error=True)
    counts=dict(sampling={key:dict(sources=n,fraction=n/total,
        occupied_core_envelope_sqdeg=sum(r['area_sqdeg'] for r in wide['cores'].values() if r['observing_support_counts'].get(key,0)>0)) for key,n in support.items()},
        observed_qso_cell_sources=sum(r['all_magnitude_source_count'] for r in cells.values()),total=total)
    report=dict(frozen_science_sha256=digest(file),quality_sha256=digest(quality),counts=counts,cells=cells,
        fields=dict(source='Keep original quality_flags, kind, physical groups, independent/reference 4x4 components and graph SHA. Sampling classes are conditional research-use labels, not stellar identity certification.',
            region='Look up native J2015.5 position in the NSIDE32 evidence cells and the reference-r layer. Do not silently rescale or subtract QSO zeros from individual sources.',
            covariance='Original conditional+reference covariance preserved. No learned model passed all identity and spatial transfer gates; no universal faint-stellar scale adopted.',
            area='Occupied core envelopes overlap between classes and are upper envelopes, not calibrated or pure-star area. NSIDE32 cells are evidence support, not full-cell certification. Geometric opportunity remains separately frozen.',
            certified_faint_star_systematic_area_sqdeg=None),slurm_job=os.environ['SLURM_JOB_ID'])
    check_aggregate(report);save_json(output/'v20_ancillary_quality_summary.json',report)
    # The original 197-object study used its original frozen graph. Verify rather
    # than assume that the expanded final reference changed nothing in that sky.
    old=read(cluster/'private_cluster_diagnostics.json.gz')
    entries={tuple(v['source_key']):v for v in old['entries']}
    # Diagnose stellar color only within fixed actual sampling strata; do not fit
    # a new color correction or use these bright stars as faint-star truth.
    color_conditions={}
    bright=[v for v in old['entries'] if v.get('gaia') and v.get('sdss_g_minus_i') is not None]
    for lower,upper in [(15,19),(19,21)]:
        for runs in (1,2):
            for lo,hi in [(-1,.5),(.5,1),(1,1.5),(1.5,2),(2,4)]:
                chosen=[v for v in bright if v['row'].get('magnitude') is not None and lower<=v['row']['magnitude']<upper and (len(v['sdss_runs'])==1)==(runs==1) and lo<=v['sdss_g_minus_i']<hi]
                if len(chosen)<20:continue
                residual=np.asarray([np.asarray(v['row']['pm_icrs_masyr'])-[v['gaia']['pmra'],v['gaia']['pmdec']] for v in chosen])
                color_conditions[f'r{lower}_{upper}__run{runs}__gi{lo}_{hi}']=dict(n=len(chosen),median=np.median(residual,axis=0),
                    nmad=1.4826*np.median(abs(residual-np.median(residual,axis=0)),axis=0),
                    distinct_sdss_groups=len({g for v in chosen for g in v['sdss_runs']}))
    save_json(output/'v20_ngc_color_condition_summary.json',dict(cohorts=color_conditions,
        scope='Bright Gaia validation only; source magnitude and single/multiple real run strata fixed. Observational color dependence, not causal attribution or adopted correction.',slurm_job=os.environ['SLURM_JOB_ID']))
    keys=old['selected']['old__target__21.0']
    previous=[entries[tuple(k)]['row'] for k in keys]
    current_rows=[]
    for owned,_,cores in footprints(current):
        for core in cores:current_rows.extend(r for _,_,r in owned_records(owned,core))
    memberships=[set(r['detection_ids']) for r in current_rows]
    owner=defaultdict(list)
    for i,member in enumerate(memberships):
        for d in member:owner[d].append(i)
    matched=[];deltas=[];exact=0;ambiguous=0;used=[]
    for row in previous:
        member=set(row['detection_ids']);votes=Counter(i for d in member for i in owner[d])
        eligible=[i for i,n in votes.items() if n>=.6*len(member) and n>=.6*len(memberships[i])]
        if len(eligible)!=1:ambiguous+=1;continue
        i=eligible[0];used.append(i);new=current_rows[i]
        exact+=member==memberships[i];matched.append(new)
        deltas.append(np.asarray(new['pm_icrs_masyr'])-row['pm_icrs_masyr'])
    if len(set(used))!=len(used):raise ValueError('Current graph counterpart is not unique')
    out=dict(n_previous=len(previous),unique_physical_counterparts=len(matched),exact_members=exact,
        ambiguous_or_absent=ambiguous,scientific_context='Same old photometric identities connected by physical detections, not matched by fitted PM.',
        slurm_job=os.environ['SLURM_JOB_ID'])
    if matched:
        weights=1/np.mean([np.asarray(r['pm_error_masyr'])**2 for r in matched],axis=1);weights/=weights.sum()
        out.update(current_weighted_mean=weights@np.asarray([r['pm_icrs_masyr'] for r in matched]),
            graph_update_pm_median=np.median(deltas,axis=0),graph_update_pm_max_abs=np.max(abs(np.asarray(deltas)),axis=0))
    check_aggregate(out);save_json(output/'v20_ngc_graph_transfer_summary.json',out)
    print('ANCILLARY_DONE',total,'NGC_COUNTERPARTS',len(matched),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('current','quality','cluster','output'):p.add_argument(name,type=Path)
    a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use Slurm resources')
    run(**vars(a))
