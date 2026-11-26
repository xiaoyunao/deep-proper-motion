"""Cross the frozen historical sky and actual SDSS-run strata; no fitting."""
import argparse
import os
from pathlib import Path

from deep_pm.blind_catalogue_quality import read
from deep_pm.full_coordinate_quality import metrics
from deep_pm.io import digest,save_json
from deep_pm.public_summary import check_aggregate
from deep_pm.stream_json import array_items
import deep_pm.full_coordinate_quality as implementation


def run(root):
    folder=root/'quality'
    source=folder/'private_final_blind_QSO.json.gz'
    main=folder/'v22_global_QSO_quality_summary.json'
    baseline=read(main)
    ticket=dict(frozen_spectral_output_sha256=digest(source),main_readout_sha256=digest(main),
        metric_implementation_sha256=digest(Path(implementation.__file__)),
        cross_readout_implementation_sha256=digest(Path(__file__)))
    output=folder/'v22_historical_run_quality_summary.json'
    receipt=folder/'private_historical_run_readout_receipt.json'
    if receipt.exists():
        done=read(receipt)
        if done['ticket']!=ticket or done['output_sha256']!=digest(output):
            raise ValueError('Changed frozen cross-stratum input')
        return
    areas=('pre_v19_2154deg','v19_new_597deg')
    groups={(low,area,single):[] for low in (21,22) for area in areas for single in (True,False)}
    for row in array_items(source):
        mag=row.get('magnitude')
        if mag is None or not 21<=mag<23:continue
        area=row['historical_area'];runs=row['sdss_runs']
        if area not in areas or runs<1:raise ValueError('Missing frozen sky/run definition')
        groups[(int(mag),area,runs==1)].append(dict(values=row['values'],pm_covariance=row['pm_covariance']))
    result={}
    for low in (21,22):
        cohorts=[]
        for area in areas:
            for single in (True,False):
                part=groups[(low,area,single)]
                cohorts.append(dict(historical_area=area,single_SDSS_run=single,spectral_denominator=len(part),
                    strict_published_common=metrics(part,('new_all','Xiao','NSC'))))
                print('HISTORICAL_RUN_READOUT',low,area,single,len(part),flush=True)
        total=sum(r['spectral_denominator'] for r in cohorts)
        common=sum(r['strict_published_common']['n'] for r in cohorts)
        expected=baseline['new_final_blind'][str(low)]['all']
        if total!=expected['spectral_denominator'] or common!=expected['strict_published_common']['n']:
            raise ValueError('Historical/run partition does not reproduce the complete frozen sample')
        result[str(low)]=dict(spectral_denominator=total,strict_common_total=common,cohorts=cohorts)
    report=dict(ticket=ticket,strata=result,slurm_job=os.environ['SLURM_JOB_ID'],
        scope='All eight predefined historical-area × actual single/multiple SDSS-run × faint-magnitude cells. Same frozen final blind identities, point estimates, quality and 800-draw paired statistic as the complete readout. No refitting, new selection or model training. Source-sampling intervals remain conditional; this is a descriptive cross-stratification of the existing footprint, not a fresh independent test area.')
    check_aggregate(report);save_json(output,report)
    save_json(receipt,dict(ticket=ticket,output_sha256=digest(output)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    args=parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):parser.error('Use Slurm')
    run(args.root)
