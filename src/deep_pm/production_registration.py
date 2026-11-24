"""Freeze the next wide batch by native metadata and full-domain replay cost."""
import argparse
from collections import Counter
from itertools import combinations
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .blind_catalogue_quality import read
from .continuous_reference import read_offsets, verify_directory
from .exposure_catalogues import inside
from .independent_selection import intersects_previous
from .io import digest, save_json
from .public_summary import check_aggregate

TARGET = 800.
MAIN_NODE_HOURS = 48.
TIB = 1024**4


def projected_cost(windows, domains):
    new = sum(w['new_cores'] for w in windows)
    raw = sum(w['input_rows_proxy'] for w in windows)
    old = set().union(*(w['overlap'].keys() for w in windows))
    replay = sum(domains[k]['cores'] for k in old)
    total = replay+new
    # Previous batch actual intake: 11.59 hours for 419.4M rows.
    # Full fit: 29.2244 hours/2251 cores. Plan only 10% of the
    # measured six-core 15% gain, and include replay, closure and margin.
    stages = dict(intake=max(.012*new, 11.59*raw/419424538),
                  reference_projection=.0011*total,
                  reference_solver=1.1+.006*total,
                  fit=(29.2244/2251)*.90*total,
                  final_owner_and_quality=1.7+.0015*total,
                  planning_and_recovery=1.3)
    peak = int(raw*1700 + new*25000000 + total*28500000)
    return dict(estimated_cpu_node_hours=sum(stages.values()), estimated_cpu_node_hours_by_stage=stages,
                estimated_additional_peak_bytes=peak,estimated_additional_inodes=new*1150+total*10,
                touched_prior_domains=sorted(old),estimated_old_replayed_cores=replay,
                expected_new_normalized_rows_with_halo=raw,expected_new_research_fits=int(5260*new))


def register(metadata, frozen, spectra, output):
    summary=frozen/'continuation_production_science_summary.json'
    receipt=read(frozen/'continuation_science_receipt.json')
    if digest(summary)!=receipt['combined_science_sha256']:
        raise ValueError('Previous frozen footprint digest changed')
    old=read(summary)
    windows=read(metadata/'wide_geometry_summary.json')['windows']
    manifest=frozen/'private_continuation_manifest.json'
    domains={}
    for part in read(manifest)['partitions']:
        owned=Path(part['owned'])
        verify_directory(owned,('fits.json','continuous_science_summary.json'))
        if digest(owned/'continuous_science_summary.json')!=receipt['graphs'][part['label']]['science_receipt_sha256']:
            raise ValueError('Prior ownership receipt changed')
        paths=read(owned/'fits.json')
        folder=Path(next(iter(paths.values())))
        shared=(folder/'reference_receipt.json').resolve().parent
        graph=shared/'private_c0.npz'
        verify_directory(shared,('private_c0.npz',))
        if digest(graph)!=receipt['graphs'][part['label']]['reference_graph_sha256']:
            raise ValueError('Prior reference graph digest changed')
        domains[part['label']]=dict(cores=len(paths),groups=set(read_offsets(graph)))
    verify_directory(spectra.parent,(spectra.name,))
    qso=pd.read_csv(spectra,usecols=['ra','dec','reference_r'])
    qso=qso.loc[qso.reference_r.between(21,23,inclusive='left')]
    eligible=[]
    for w in windows:
        if not w['selected_parents']:
            continue
        name=w['name']
        if name in old['regions']:
            raise ValueError('Candidate window overlaps a processed window')
        possible=set(read(metadata/name/'private_possible_groups.json'))
        overlap={k:len(possible&d['groups']) for k,d in domains.items() if possible&d['groups']}
        qa=np.zeros(len(qso),bool)
        for parent in w['selected_parents']:
            box=dict(ra=parent['ra_bounds_deg'],dec=parent['dec_bounds_deg'])
            qa|=inside(qso.ra,qso.dec,box)
        new=4*len(w['selected_parents'])
        # Broad early estimate. Exposure-support proxy changes with repeated
        # NSC depth; dense cores remain in the first acquisition tranche.
        exposure_total=sum(t['exposures'] for t in w['selected_parents'])
        proxy=int(max(new*310000,exposure_total*5200))
        rich=sum(t['area_sqdeg'] for t in w['selected_parents']
                 if t['nsc_window_span_years']>=4 and t['nsc_window_calendar_years']>=3)
        eligible.append(dict(window=w,new_cores=new,input_rows_proxy=proxy,overlap=overlap,
                             qso_21_23_position_count=int(qa.sum()),metadata_rich_area_sqdeg=rich))
    eligible.sort(key=lambda w:w['window']['name'])
    # Keep whole candidate windows. Evaluate subsets of each permitted prior-domain
    # combination by a greedy cadence/coverage tie-break while charging old cores once.
    best=None
    for n in range(len(domains)+1):
        for allowed in combinations(domains,n):
            items=[w for w in eligible if set(w['overlap'])<=set(allowed)]
            if not items:
                continue
            for weight in (0.,.7,1.5):
                ordered=sorted(items,key=lambda w:(
                    -(w['window']['pre_reference_opportunity_sqdeg']+weight*w['metadata_rich_area_sqdeg'])
                     /(1+w['input_rows_proxy']/1e8),
                    w['window']['name']))
                chosen=[]
                for w in ordered:
                    trial=chosen+[w]
                    nominal=sum(x['window']['suitable_nominal_sqdeg'] for x in trial)
                    cost=projected_cost(trial,domains)
                    if nominal>1000 or cost['estimated_cpu_node_hours']>MAIN_NODE_HOURS or cost['estimated_additional_peak_bytes']>TIB:
                        continue
                    chosen=trial
                    if nominal>=TARGET:
                        break
                nominal=sum(x['window']['suitable_nominal_sqdeg'] for x in chosen)
                if nominal<500:
                    continue
                cost=projected_cost(chosen,domains)
                credible=sum(w['window']['pre_reference_opportunity_sqdeg'] for w in chosen)
                score=credible-.65*abs(nominal-TARGET)+.1*sum(w['metadata_rich_area_sqdeg'] for w in chosen)
                if best is None or score>best[0]:
                    best=(score,chosen,cost)
    if best is None:
        raise ValueError('No 500–1000 square degree complete-window combination fits the measured budget proxy')
    _, selected, cost=best
    chosen=[r for w in selected for r in w['window']['selected_parents']]
    previous=[dict(ra=r['core_bounds_deg']['ra_bounds_deg'],dec=r['core_bounds_deg']['dec_bounds_deg'])
              for r in old['cores'].values()]
    active=[]
    for r in chosen:
        box=dict(ra=r['ra_bounds_deg'],dec=r['dec_bounds_deg'])
        if intersects_previous(box,previous+active):
            raise ValueError('Registered new geometry overlaps the prior footprint or another parent')
        active.append(box)
    nominal=sum(r['area_sqdeg'] for r in chosen)
    output.mkdir(parents=True,exist_ok=True)
    section=['nsc_all_objects = true','reference_epoch = 2015.5','nside = 128','halo_deg = 0.06','seed = 9137']
    windows_public={}
    for w in selected:
        v=w['window'];name=v['name']; b=v['window_bounds_deg']
        section.extend([f'[boxes.{name}]',f'ra = {b["ra_bounds_deg"]}',f'dec = {b["dec_bounds_deg"]}'])
        windows_public[name]=dict(area_sqdeg=v['suitable_nominal_sqdeg'],cores=w['new_cores'],
                        qso_21_23_position_count=w['qso_21_23_position_count'],
                        metadata_rich_area_sqdeg=w['metadata_rich_area_sqdeg'],
                        native_sdss_multi_fraction_across_window=v['geometry_sdss_multi_fraction'],
                        candidate_shared_physical_groups_by_prior_domain=w['overlap'])
    protocol=output/'next_windows.toml'
    txt='\n'.join(section)+'\n'
    if protocol.exists() and protocol.read_text()!=txt:
        raise ValueError('Already frozen a different production protocol')
    protocol.write_text(txt)
    answer=dict(first_batch=chosen,windows=sorted(windows_public),window_metadata=windows_public,
                new_nominal_area_sqdeg=nominal,new_cores=4*len(chosen),
                nominal_area_if_completed_sqdeg=nominal+sum(r['area_sqdeg'] for r in old['cores'].values()),
                pre_reference_opportunity_sqdeg=sum(r['pre_reference_opportunity_sqdeg'] for r in chosen),
                budget_nodes_hours=MAIN_NODE_HOURS,estimated_resources=cost,
                previous_scope_sha256=digest(frozen/'continuation_science_receipt.json'),
                geometry_sha256=digest(metadata/'wide_geometry_summary.json'),config_sha256=digest(protocol),
                spectral_metadata_sha256=digest(spectra),slurm_job=os.environ['SLURM_JOB_ID'],
                scope='Whole qualified-window selection from native footprint and cadence before new proper motion outcomes. Window cadence and possible group intersections are planning proxies. Exact replay scope comes from signed acquired reference galaxies.')
    check_aggregate(answer)
    save_json(output/'wide_selection_summary.json',answer)
    print('Registered',len(windows_public),len(chosen)*4,nominal,cost,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('metadata','frozen','spectra','output'):
        p.add_argument(arg,type=Path)
    args=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        p.error('Use Slurm compute node')
    register(**vars(args))
