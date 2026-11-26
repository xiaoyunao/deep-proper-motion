"""Read existing signed receipts without changing fits or selection.

Origin-core funnels and per-component applications have distinct denominators.
They describe where selection occurs; the combined coordinate/reference update
is not a factorial experiment isolating a single cause of every changed source.
"""
import argparse
from collections import Counter,defaultdict
import os
from pathlib import Path

import numpy as np

from .blind_catalogue_quality import read
from .full_coordinate_catalogue import database
from .io import digest,save_json
from .public_summary import check_aggregate


def add_counts(target,value,prefix=''):
    for key,number in value.items():
        name=prefix+'/'+key if prefix else key
        if isinstance(number,dict):add_counts(target,number,name)
        elif isinstance(number,int) and not isinstance(number,bool):target[name]+=number
        else:raise ValueError('Selection funnel must contain integer counts only')


def aggregate(proofs):
    counts=Counter();funnel=Counter();support=defaultdict(Counter);elapsed=[];rss=[]
    for proof in proofs:
        for key in ('research_solutions','r_control_solutions','physical_r_control_solutions'):
            counts[key]+=proof[key]
        for key,number in proof['by_class'].items():counts['class__'+key]+=number
        for component,entries in proof['funnel'].items():add_counts(funnel,entries)
        for component,values in proof['support'].items():
            for key in ('core_group_present','core_local12mas_supported'):support[component][key]+=values[key]
        elapsed.append(proof['elapsed_seconds']);rss.append(proof['peak_rss_mib'])
    return dict(origin_counts=dict(counts),selection_funnel=dict(funnel),component_detection_applications={k:dict(v) for k,v in support.items()},
        core_count=len(proofs),summed_source_worker_elapsed_seconds=sum(elapsed),
        core_elapsed_seconds_p16_p50_p84=np.quantile(elapsed,[.16,.5,.84]).tolist() if elapsed else None,
        maximum_source_worker_elapsed_seconds=max(elapsed) if elapsed else None,
        source_workers_elapsed_ge600_seconds=sum(v>=600 for v in elapsed),
        source_workers_elapsed_ge1800_seconds=sum(v>=1800 for v in elapsed),
        maximum_source_process_rss_mib=max(rss) if rss else None,
        core_rss_mib_p16_p50_p84=np.quantile(rss,[.16,.5,.84]).tolist() if rss else None)


def run(root):
    plan=read(root/'registration/private_full_coordinate_plan.json');proofs=[];signatures=[]
    for entry in plan['cores']:
        folder=root/'sources'/entry['core'];path=folder/'v22_source_core_summary.json';proof=read(path)
        # The final compact index already checked each private source hash.
        # Bind the readout to exactly that scientific receipt, not a fresh fit.
        index=read(root/'catalogue/index'/entry['core']/'private_index_receipt.json')
        if index['ticket']['source_summary_sha256']!=digest(path):raise ValueError('Source and accepted index versions differ')
        proofs.append(proof);signatures.append(digest(path))
    report=aggregate(proofs);report['source_receipt_set_sha256']=digest_text('\n'.join(signatures))
    db=database(root)
    report['accepted_observation_flags']=dict(db.execute('''SELECT
        count(*) all_accepted,
        count(*) FILTER(WHERE maximum_group_leverage>=.8) high_physical_group_leverage,
        count(*) FILTER(WHERE nsc_seasons<3) fewer_than_three_nsc_seasons,
        count(*) FILTER(WHERE sdss_runs=1) single_physical_SDSS_run,
        count(*) FILTER(WHERE sdss_runs>1) multiple_physical_SDSS_runs,
        count(*) FILTER(WHERE flags LIKE '%unknown_or_invalid_morphology%') unknown_or_invalid_morphology,
        count(*) FILTER(WHERE flags LIKE '%spatial_zero_uncertified%') spatial_zero_uncertified,
        count(*) FILTER(WHERE flags LIKE '%night_correlation_unqualified%') night_correlation_unqualified
        FROM accepted''').fetchdf().iloc[0].items())
    query='''SELECT kind,sdss_runs=1 single_run,count(*) source_count,
        quantile_cont(sqrt(independent_pm_variance_e),[.16,.5,.84]) independent_E_SE,
        quantile_cont(sqrt(independent_pm_variance_n),[.16,.5,.84]) independent_N_SE,
        quantile_cont(sqrt(reference_pm_variance_e),[.16,.5,.84]) reference_E_SE,
        quantile_cont(sqrt(reference_pm_variance_n),[.16,.5,.84]) reference_N_SE,
        quantile_cont(nsc_nights,[.16,.5,.84]) NSC_nights,
        quantile_cont(nsc_seasons,[.16,.5,.84]) NSC_seasons,
        quantile_cont(joint_span,[.16,.5,.84]) joint_span_years
        FROM accepted GROUP BY kind,single_run'''
    report['accepted_source_condition_groups']=db.execute(query).fetchdf().to_dict('records')
    report['accepted_source_r_magnitude_groups']=db.execute('''SELECT
        CASE WHEN magnitude IS NULL OR NOT isfinite(magnitude) THEN 'missing'
             WHEN magnitude<20 THEN 'r_lt20' WHEN magnitude<21 THEN 'r_20_21'
             WHEN magnitude<22 THEN 'r_21_22' WHEN magnitude<23 THEN 'r_22_23'
             ELSE 'r_ge23' END magnitude_bin,
        kind,sdss_runs=1 single_run,count(*) source_count
        FROM accepted GROUP BY magnitude_bin,kind,single_run''').fetchdf().to_dict('records')
    if sum(r['source_count'] for r in report['accepted_source_r_magnitude_groups'])!=report['accepted_observation_flags']['all_accepted']:
        raise ValueError('Source magnitude groups do not close the accepted catalogue')
    report['source_magnitude_definition']='Actual adopted r magnitude: NSC r median where available, otherwise SDSS r. These source-count bins are not the independent spectroscopic r bins of the QSO comparisons; neither measures a photometric-parent completeness.'
    report['readout_implementation_sha256']=digest(Path(__file__))
    db.close()
    report['denominator_notes']=(
        'Origin counts and funnel stages are before common-epoch transfer/global identity exclusions. '
        'Component detection applications count normalized core positions, not globally unique measurements; '
        'do not pool across disconnected component applications. Anchors enter the r funnel after morphology '
        'and core selection, and stage counters overlap; they are not an all-photometric parent. '
        'No new selection, fit, DCR or error rescaling. Native support, spherical association and weighting '
        'changed together in the full update; these receipt counts do not causally isolate their source gains. '
        'The fixed v2.1 controlled experiments and v2.2 fixed-identity comparisons supply the separate tests. '
        'Source-worker elapsed is summed worker walltime, not CPU or allocation billing. '
        'Independent/reference SEs are conditional mas/yr, not absolute systematic uncertainties. '
        'Quality flags overlap and do not certify stellar identity or spatial zero.')
    report['slurm_job']=os.environ['SLURM_JOB_ID']
    check_aggregate(report);save_json(root/'catalogue/v22_production_selection_summary.json',report)


def digest_text(value):
    from hashlib import sha256
    return sha256(value.encode()).hexdigest()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use Slurm')
    run(a.root)
