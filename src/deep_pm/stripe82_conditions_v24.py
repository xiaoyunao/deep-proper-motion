"""Prespecified magnitude-by-scan readout of the signed common HLC star sample."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom

from .io import digest,save_json
from .stripe82_comparison_v24 import EDGES,METHODS,binomial,stats


def median_interval(values):
    # Distribution-free order-statistic interval, conditional on objects;
    # the separately reported physical-run sensitivity captures shared inputs.
    v=np.sort(np.asarray(values))
    if len(v)<6:return None
    k=int(binom.ppf(.025,len(v),.5))
    return [float(v[max(0,k-1)]),float(v[min(len(v)-1,len(v)-k)])]


def readout(directory):
    source=directory/"private_pairs.parquet"
    marker=directory/"v24_stripe82_comparison_summary.json"
    receipt=json.loads(marker.read_text())
    if digest(source)!=receipt["pair_sha256"]:raise ValueError("Changed signed HLC comparison")
    columns=["identity","in_nominal","time_opportunity","r","colour","neighbour","sdss_runs","sdss_years",
             "nsc_span","joint_span","pm_e","pm_n",*[m+"_"+a for m in METHODS for a in ("e","n")]]
    table=pd.read_parquet(source,columns=columns)
    parent=table.loc[table.in_nominal & table.time_opportunity]
    common=parent.dropna(subset=[m+"_"+a for m in METHODS for a in ("e","n")])
    if len(common)!=receipt["total"]["common"] or len(parent)!=receipt["in_nominal_time"]:
        raise ValueError("HLC population changed during condition readout")
    run_conditions={"one_run":common.sdss_runs==1,"multiple_runs":common.sdss_runs>=2}
    if sum(int(g.sum()) for g in run_conditions.values())!=len(common):raise ValueError("Unknown scan count")
    bins={};deep=None
    for lo,hi in zip(EDGES,EDGES[1:]):
        magnitude=(common.r>=lo)&(common.r<hi)
        parents=int(((parent.r>=lo)&(parent.r<hi)).sum())
        n=int(magnitude.sum())
        cell={"common_in_hlc_opportunity":binomial(n,parents),"by_sdss_run":{}}
        if n>=20:deep=[lo,hi,n]
        for title,mask in run_conditions.items():
            selected=common.loc[magnitude&mask]
            result={m:stats(selected,m) for m in METHODS}
            if len(selected):
                for m in METHODS:
                    result[m]["median_object_conditional_95"]=[median_interval(selected[m+"_"+a]-selected["pm_"+a]) for a in ("e","n")]
            cell["by_sdss_run"][title]=dict(common=len(selected),relative_to_hlc=result,
                distinct_sdss_year_median=(float(selected.sdss_years.median()) if len(selected) else None))
        bins[f"{lo:g}-{hi:g}"]=cell
    save_json(directory/"v24_stripe82_conditional_summary.json",dict(scope=receipt["scope"],pair_sha256=receipt["pair_sha256"],
              faintest_magnitude_bin_with_20_common=deep,bins=bins,
              dependence="HLC and all three ground estimators can share inputs. These are observed differences. "
              "Run counts pertain to our adopted fit. Scan-stratified common counts do not establish a recovery fraction "
              "for HLC parents lacking a blind ground fit."))


def scan_sensitivity(directory):
    proof=json.loads((directory/"v24_stripe82_scan_ablation_summary.json").read_text())
    pair=directory/"private_run_ablation_results.parquet"
    parent=directory/"private_pairs.parquet"
    original=json.loads((directory/"v24_stripe82_comparison_summary.json").read_text())
    if digest(pair)!=proof["private_result_sha256"] or digest(parent)!=original["pair_sha256"]:
        raise ValueError("Changed ablation or identity input")
    frame=pd.read_parquet(pair).merge(pd.read_parquet(parent,columns=["identity","origin_core"]),on="identity",validate="one_to_one")
    if len(frame)!=proof["rows"]:raise ValueError("Missing paired star")
    rng=np.random.default_rng(240082)
    stats={}
    for lo in (21,22):
        group=frame[frame.r.between(lo,lo+1,inclusive="left")]
        rad=np.array([np.hypot(group[m+"_e"]-group.pm_e,group[m+"_n"]-group.pm_n) for m in ("all","first")])
        if len(group)<2:continue
        def difference(ix):
            a,b=np.quantile(rad[:,ix],.68,axis=-1)
            return a-b
        resampled=rng.integers(0,len(group),size=(2000,len(group)))
        leave=[float(difference(group.origin_core.to_numpy()!=key)) for key in sorted(group.origin_core.unique())
               if (group.origin_core!=key).any()]
        stats[str(lo)]=dict(n=len(group),difference_all_minus_earliest=float(difference(np.arange(len(group)))),
                            paired_object_interval=np.quantile(difference(resampled),[.025,.975]),
                            actual_leave_one_core_range=[min(leave),max(leave)] if leave else None,cores=group.origin_core.nunique())
    save_json(directory/"v24_stripe82_scan_sensitivity_summary.json",dict(scope=proof["scope"],by_magnitude=stats,
              interpretation="Paired radial P68 difference (mas/yr). Intervals resample fixed stars conditional on the fields. "
              "Leave-one-core sensitivity includes shared early observations and has few regions."))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("directory",type=Path);p.add_argument("--scan-sensitivity",action="store_true");a=p.parse_args()
    if not os.getenv("SLURM_JOB_ID"):p.error("Use assigned Slurm CPUs")
    if a.scan_sensitivity:scan_sensitivity(a.directory)
    else:readout(a.directory)
