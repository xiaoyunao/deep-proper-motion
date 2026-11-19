"""Trace whether rejected zero-point errors are common translations or relative errors."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .continuous_reference import bind,deduplicate,verify_directory
from .io import save_json,digest
from .provenance import run_stage
from .public_summary import check_aggregate


def counts_after_reference_removal(train,removed):
    sub=train.loc[~train.reference.isin(removed)]
    r,refs=pd.factorize(sub.reference);g,groups=pd.factorize(sub.group)
    graph=coo_matrix((np.ones(len(sub)),(r,len(refs)+g)),shape=(len(refs)+len(groups),)*2)
    _,label=connected_components(graph,directed=False)
    return sorted(Counter(label[len(refs):]).values(),reverse=True)


from collections import Counter


def run(work,reference,plan):
    table=pd.read_csv(reference/'private_reference_group_selection.csv.gz')
    rejected=set(table.loc[table.status.eq('offset_se_rejected'),'group'])
    retained=set(table.loc[table.status.eq('retained'),'group'])
    with np.load(reference/'private_c0_unfiltered.npz',allow_pickle=False) as a:
        select=np.isin(a['names'],sorted(rejected));draw=a['draws'][select]
        median=np.median(draw,axis=0)
        original=np.std(draw,axis=1,ddof=1);relative=np.std(draw-median[None],axis=1,ddof=1)
        diagnostics=dict(rejected_group_count=int(select.sum()),
            marginal_se_p16_p50_p84_mas=np.quantile(original,[.16,.5,.84],axis=0),
            cluster_median_common_translation_se_mas=np.std(median,axis=0,ddof=1),
            relative_to_cluster_translation_se_p16_p50_p84_mas=np.quantile(relative,[.16,.5,.84],axis=0),
            cluster_translation_covariance_mas2=np.cov(median,rowvar=False),
            scope='Synchronous bootstrap decomposition only; subtracting a common translation here is diagnostic, not an adopted removal of absolute position uncertainty.')
    frames=[pd.read_csv(Path(p)/'reference.csv.gz',dtype={'reference':str,'is_holdout':bool}) for p in json.loads(plan.read_text()).values()]
    rows,selection=deduplicate(frames);train=rows.loc[~rows.is_holdout].copy()
    train['retained']=train.group.isin(retained);train['rejected']=train.group.isin(rejected)
    touch=train.groupby('reference')[['retained','rejected']].any()
    bridges=set(touch.index[touch.retained&touch.rejected]);bridge=train.loc[train.reference.isin(bridges)]
    within_rejected=train.loc[train.rejected]
    diagnostic_counts=table.loc[table.group.isin(rejected)]
    result=dict(bootstrap_modes=diagnostics,rejected_training_reference_count=within_rejected.reference.nunique(),
        rejected_group_training_count_p16_p50_p84=np.quantile(diagnostic_counts.training_reference_count,[.16,.5,.84]),
        connecting_reference_count=len(bridges),connecting_reference_measurement_count=len(bridge),
        connecting_reference_groups=bridge.group.nunique(),
        connecting_measurement_sigma_p16_p50_p84_mas=np.quantile(bridge[['sigma_e','sigma_n']],[.16,.5,.84],axis=0) if len(bridge) else None,
        connected_group_sizes_without_connecting_references=counts_after_reference_removal(train,bridges),
        reference_deduplication=selection,slurm_job=os.environ['SLURM_JOB_ID'],
        scope='All original training references and same model; no new data, no QSO tuning, no support restored. Counts show physical links between retained and SE-rejected subgraphs. Full reference covariance must be retained if support criteria are later tested.')
    check_aggregate(result);save_json(work/'reference_support_modes_summary.json',result)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('work','reference','plan','config'):p.add_argument(n,type=Path)
    a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use a Slurm compute node')
    a.work.mkdir(parents=True,exist_ok=True)
    bind(a.work,'diagnostic_reference_receipt.json',verify_directory(a.reference,('private_c0_unfiltered.npz','private_reference_group_selection.csv.gz')))
    inputs=['diagnostic_reference_receipt.json']
    for name,path in json.loads(a.plan.read_text()).items():
        alias=name+'_receipt.json';bind(a.work,alias,verify_directory(Path(path),('reference.csv.gz',)));inputs.append(alias)
    with run_stage(a.work,a.config,'a','reference-support-modes-v14',inputs=inputs,outputs=('reference_support_modes_summary.json',)) as go:
        if go:run(a.work,a.reference,a.plan)


if __name__=='__main__':main()
