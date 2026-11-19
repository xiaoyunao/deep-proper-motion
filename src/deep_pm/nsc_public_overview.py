"""Compact, source-hashed review copy of the complete public v1.5 aggregates."""
import argparse
import json
from pathlib import Path

from .io import digest,save_json
from .public_summary import check_aggregate


def compact(sample):
    fields=('radial_abs68','median','median_bias95','nmad','fraction_within_1sigma','fraction_within_2sigma')
    return dict(n=sample['n'],methods={m:{k:v[k] for k in fields if k in v} for m,v in sample.get('methods',{}).items()},paired=sample.get('paired',{}))


def run(directory,output):
    main=directory/'v15_continuous_science_summary.json';geopath=directory/'v15_geographic_readout_summary.json';data=json.loads(main.read_text());geo=json.loads(geopath.read_text());known={};geographic={}
    for low in (21,22):
        for tag in ('all','one_sdss_run','multiple_sdss_runs','one_run_pointlike'):
            key=f'r{low}__{tag}';sample=data['known_stationary'][key];known[key]=dict(common=compact(sample['strict_published_common']),connected_blocks=sample['observational_blocks']['physical']['independent_blocks'],shared_group_sensitivity=sample['shared_group_sensitivity'])
        for region in ('ordinary','transition','repeat_band'):
            sample=geo['cohorts'][region]['r'+str(low)];geographic[f'{region}__r{low}']=dict(common=compact(sample['common']),shared_group_sensitivity=sample['shared_group_sensitivity'])
    result=dict(area_sqdeg=data['completed_nominal_core_area_sqdeg'],completed_cores=data['completed_cores'],research_counts={k:data[k] for k in ('research_r_control_solutions','research_physical_r_solutions','research_all_nsc_solutions')},all_magnitude_source_classes=data['all_magnitude_source_classes'],by_class={k:v['research_fits'] for k,v in data['by_class'].items()},
        known_precision=known,original_geographic_definition=geo['scope'],geographic_precision=geographic,blind_spectral={k:compact(v['all_spectral']) for k,v in data['blind_spectral'].items()},source_quality_flag_counts=data['source_quality_flag_counts'],ownership=data['ownership'],reference_graph_sha256=data['reference_graph_sha256'],
        source_summary_sha256=digest(main),geographic_summary_sha256=digest(geopath),scope='Review copy only; exact existing aggregate values. No new fit, resampling, sample selection or certification. Single SDSS run differs from the original geographic ordinary band; preserve both. Complete spatial, information, coverage and band diagnostics remain in the full source files. Research counts are not purity-certified stars; source/region systematics and identity limitations remain explicit.')
    check_aggregate(result);save_json(output,result)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);p.add_argument('output',type=Path);a=p.parse_args();run(a.directory,a.output)


if __name__=='__main__':main()
