"""Summarize owned research catalogues across compatible regional C0 graphs."""
import argparse
from collections import Counter, defaultdict
import gzip
import json
import os
from pathlib import Path

import healpy as hp
import numpy as np

from .__main__ import tomllib
from .continuous_summary import area
from .io import digest, json_default, save_json
from .nsc_comparison import comparison
from .nsc_coverage_conditions import coverage
from .nsc_ownership_overlay import owned_records
from .nsc_science_summary import SETTINGS, known_cohorts
from .public_summary import check_aggregate


def condition_readout(rows):
    return coverage([dict(core=row['core'], sdss=[v for v in row['exposures'] if v.startswith('SDSS:')],
                          nights=len(row['nsc_epoch_days']), span=row['joint_span'], **row['values'])
                     for row in rows])


def summarize(known, blind):
    tests = known_cohorts(known, instrument_diagnostics=True)
    out = dict(known_stationary=tests, formal_error_coverage={}, blind_spectral={})
    for low in (21, 22):
        common = [r for r in known if low<=r['magnitude']<low+1 and {'Xiao','NSC'}<=r['values'].keys()]
        for tag,predicate in (('all', lambda r:True), ('joint_lt12', lambda r:r['joint_span']<12),
                              ('joint_ge12', lambda r:r['joint_span']>=12)):
            out['formal_error_coverage'][f'r{low}__{tag}'] = condition_readout([row for row in common if predicate(row)])
        for kind in ('r_existing','extra_r_anchor','non_r_anchor'):
            sample = [r for r in blind if r['magnitude'] is not None and low<=r['magnitude']<low+1 and r['kind']==kind]
            out['blind_spectral'][f'{kind}__r{low}'] = comparison([{'blind_all':r['values']['blind_all']} for r in sample], SETTINGS)
    return out


def run(manifest, output, index):
    config = json.loads(manifest.read_text())
    work = output.parent
    work.mkdir(parents=True, exist_ok=True)
    known = defaultdict(list)
    blind = defaultdict(list)
    output_by_core = {}
    pixels = defaultdict(list)
    totals = Counter()
    for part in config['partitions']:
        owned = Path(part['owned'])
        graph = json.loads((owned/'continuous_science_summary.json').read_text())
        spectral = json.loads(Path(part['private_pool']).read_text())
        if spectral['fit_receipt_sha256'] != graph['fit_receipt_sha256']:
            raise ValueError('The owned catalogue and pooled spectral data use different fits')
        specs = tomllib.loads(Path(part['geometry']).read_text())['fields']
        regions = part['region_by_core']
        if set(regions) != set(graph['spatial']):
            raise ValueError('Incomplete region assignment')
        for row in spectral['known']:
            known[regions[row['core']]].append(row)
        for row in spectral['blind']:
            blind[regions[row['core']]].append(row)
        for core, summary in graph['spatial'].items():
            if core in output_by_core:
                raise ValueError('Core occurs in more than one partition')
            members = list(owned_records(owned, core))
            if len(members) != summary['owned_all_nsc']:
                raise ValueError('Owned source count does not match the science receipt')
            conditions = Counter(flag for _,_,row in members for flag in row['quality_flags'])
            kinds = Counter(row['kind'] for _,_,row in members)
            totals.update(kinds)
            spec = specs[core]
            pixel = hp.ang2pix(128, *np.array([row['position_deg'] for _,_,row in members]).T, lonlat=True) if members else []
            for pix,(origin,slot,_) in zip(pixel,members):
                pixels[int(pix)].append((part['label'],origin,slot))
            r21 = summary['known_stationary']['r21__all']['strict_published_common']
            r22 = summary['known_stationary']['r22__all']['strict_published_common']
            output_by_core[core] = dict(
                region=regions[core], core_bounds_deg=dict(ra_bounds_deg=spec['ra'],dec_bounds_deg=spec['dec']),
                area_sqdeg=area(spec), research_fits=len(members), physical_r_fits=summary['funnel']['strict_r_solutions'],
                source_classes=dict(kinds), quality_flag_counts=dict(conditions), common_r21=r21, common_r22=r22,
                reference_component=summary['reference_gauge'],
                reference_component_core_detection_counts=summary.get('reference_component_core_detection_counts',{}),
                nside128_counts=dict(Counter(map(str,pixel))))
    region_report={}
    for region in sorted(set(c['region'] for c in output_by_core.values())):
        subset=[c for c in output_by_core.values() if c['region']==region]
        counts=Counter()
        for core in subset:
            counts.update(core['source_classes'])
        footprint = config.get('region_footprints', {}).get(region)
        opportunities = None
        if footprint:
            count = json.loads(Path(footprint).read_text())['grids']['0.025']
            opportunities = {layer:values['geometric_joint_opportunity_area_sqdeg']
                             for layer, values in count['layers'].items()}
        region_report[region]=dict(calibrated_geometric_opportunity_sqdeg=opportunities, nominal_area_sqdeg=sum(c['area_sqdeg'] for c in subset),
            source_occupied_core_area_sqdeg=sum(c['area_sqdeg'] for c in subset if c['research_fits']>0),
            research_fits=sum(c['research_fits'] for c in subset),physical_r_fits=sum(c['physical_r_fits'] for c in subset),
            source_classes=dict(counts),**summarize(known[region],blind[region]))
    row_count=sum(len(p) for p in pixels.values())
    if row_count!=sum(totals.values()):
        raise ValueError('HEALPix organization lost catalogue rows')
    with gzip.open(index, 'wt') as stream:
        json.dump(dict(nside=128, ordering='RING', partition_manifest_sha256=digest(manifest), pixels=pixels),
                  stream, default=json_default, allow_nan=False)
    result=dict(cores=output_by_core,regions=region_report,source_classes=dict(totals),
        aggregate=summarize([r for v in known.values() for r in v],[r for v in blind.values() for r in v]),
        output_healpix_pixels=len(pixels),research_fits=row_count,healpix_index_sha256=digest(index),
        slurm_job=os.environ['SLURM_JOB_ID'],
        scope='Same individual spectral objects are pooled across processing areas. Per-owner research source count, band class and flags are separate from input-frame spectral cohorts. Area with a research source is a rectangular-core coverage summary. Geometric calibrated opportunity and formal area qualification are distinct.')
    check_aggregate(result)
    save_json(output, result)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','output','index'):
        parser.add_argument(name,type=Path)
    args=parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        parser.error('Use a Slurm compute node')
    run(**vars(args))


if __name__=='__main__':
    main()
