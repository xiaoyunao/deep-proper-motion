"""Summarize C0 trial coverage, common catalogues and visit-level dependence."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from .__main__ import tomllib
from .continuous_reference import verify_directory
from .io import save_json
from .public_summary import check_aggregate
from .validation import ratio
from .validation_blocks import connected

METHODS=("NSC","Xiao","C0")
VISIT_FAMILIES=("sdss_run","nsc_exposure","nsc_night")
SAMPLING_COLUMNS=("sdss_runs","sdss_nights","sdss_span_years","nsc_nights",
                  "nsc_span_years","joint_span_years","independent_groups",
                  "pm_error_east","pm_error_north","max_group_slope_fraction")


def area(box):
    width=np.ptp(box["ra"])
    return float(np.degrees(np.ptp(np.sin(np.deg2rad(box["dec"]))))*width)


def group_labels(row, kind):
    visits = row["exposures"]
    if kind=="sdss_run":
        return [v for v in visits if v.startswith("SDSS:")]
    if kind=="nsc_exposure":
        return [v for v in visits if not v.startswith("SDSS:")]
    return row["nsc_epoch_days"]


def membership(records, kind):
    labels={}
    column=[]
    row=[]
    weights=[]
    for i,r in enumerate(records):
        tags=sorted(set(group_labels(r,kind)))
        for tag in tags:
            column.append(labels.setdefault(tag,len(labels)))
            row.append(i)
            weights.append(1/len(tags))
    matrix=csr_matrix((weights,(row,column)),shape=(len(records),len(labels)))
    return matrix


def weighted_draws(values, membership_matrix, seed):
    """Use the same weights for all methods in a paired draw."""
    order=np.argsort(values,axis=0)
    quantile=np.tile([.5,.5,.68],values.shape[1]//3)
    random=np.random.default_rng(seed)
    results=np.empty((800,values.shape[1]))
    for start in range(0,len(results),40):
        weights=membership_matrix @ random.exponential(size=(membership_matrix.shape[1],40))
        for j, selected in enumerate(order.T):
            cum=weights[selected].cumsum(axis=0)
            index=(cum>=cum[-1:]*quantile[j]).argmax(axis=0)
            results[start:start+40,j]=values[selected[index],j]
    return results.reshape(-1,values.shape[1]//3,3)


def estimate(records, methods, seed, candidate="C0"):
    if len(records)<5:
        return {"n":len(records)}
    vectors=np.array([[v["motion"][m] for m in methods] for v in records])
    radial=np.linalg.norm(vectors,axis=2)
    p68=np.quantile(radial,.68,axis=0)
    results={}
    for k,m in enumerate(methods):
        value=vectors[:,k]
        median=np.median(value,axis=0)
        results[m]=dict(median=median,nmad=1.4826*np.median(abs(value-median),axis=0),
                        radial_p68=p68[k],tail_gt_5=int(np.count_nonzero(radial[:,k]>5)),
                        visit_weighted_zero={})
    c0=methods.index(candidate)
    baseline=[(i,m) for i,m in enumerate(methods) if i!=c0]
    gains={m:dict(n=len(records),relative_gain=1-p68[c0]/p68[i] if p68[i]>0 else None,
                  visit_weighted={}) for i,m in baseline}
    values=np.concatenate((vectors,radial[:,:,None]),axis=2).reshape(len(records),-1)
    for kind in VISIT_FAMILIES:
        matrix=membership(records,kind)
        groups=matrix.shape[1]
        supported=np.diff(matrix.indptr)>0
        n=int(supported.sum())
        load=np.asarray(matrix.sum(axis=0)).ravel()
        info=float(load.sum()**2/(load@load)) if groups else 0.
        diagnostics=dict(groups=groups,n=n,equal_weight_visit_information=info)
        for m in methods:
            results[m]["visit_weighted_zero"][kind]=dict(diagnostics)
        for _,m in baseline:
            gains[m]["visit_weighted"][kind]=dict(diagnostics)
        if groups<3 or n<5:
            continue
        draws=weighted_draws(values[supported],matrix[supported],seed)
        for i,m in enumerate(methods):
            results[m]["visit_weighted_zero"][kind]["weighted_zero95"]=np.quantile(
                draws[:,i,:2],[.025,.975],axis=0).T
        for i,m in baseline:
            valid=draws[:,i,2]>0
            outcome=gains[m]["visit_weighted"][kind]
            outcome["valid_draws"]=int(valid.sum())
            if valid.any():
                outcome["relative_gain95"]=np.quantile(1-draws[valid,c0,2]/draws[valid,i,2],[.025,.975])
    return dict(n=len(records),methods=results,connected_sizes=sorted(map(len,connected(records)),reverse=True),
                shared_night_sizes=sorted(map(len,connected(records,True)),reverse=True),
                paired_radial_gain=gains)


def deletes(rows):
    affected=defaultdict(list)
    n=Counter()
    for v in rows:
        for kind,visit in v["removed"].items():
            for k,choice in visit.items():
                n[kind]+=1
                if choice is not None:
                    affected[kind].append(np.linalg.norm(np.asarray(choice)-v["baseline"]))
    return {kind:dict(attempted=int(count), remained=len(affected[kind]),
                      shift_p16_p50_p84_masyr=np.quantile(affected[kind],[.16,.50,.84])
                      if affected[kind] else None)
            for kind,count in n.items()}


def scan_strata(records):
    strata=defaultdict(list)
    for row in records:
        visits=len(set(group_labels(row,"sdss_run")))
        key="no_sdss_run" if visits==0 else "one_sdss_run" if visits==1 else "multiple_sdss_runs"
        strata[key].append(row)
    return {name:estimate(rows,METHODS,9137) for name,rows in sorted(strata.items())}


def summarize_sampling(samples):
    result={}
    for name,bin_rows in samples.groupby("r_bin",sort=True):
        by_run={}
        for label, sub in (("all",bin_rows),("one_sdss_run",bin_rows.loc[bin_rows.sdss_runs==1]),
                           ("multiple_sdss_runs",bin_rows.loc[bin_rows.sdss_runs>=2])):
            values=sub.loc[:,list(SAMPLING_COLUMNS)].to_numpy(dtype=float)
            finite=np.isfinite(values)
            by_run[label]=dict(n=len(sub),n_valid=finite.sum(axis=0),
                               p16_p50_p84=[np.quantile(v[keep],[.16,.50,.84]) if keep.any() else None
                                            for v,keep in zip(values.T,finite.T)])
        result[name]=dict(n=len(bin_rows),repeated_sdss=ratio(by_run["multiple_sdss_runs"]["n"],len(bin_rows)),
                          by_sdss_run_count=by_run)
    return result


def pool(specs, fits, shared, site_names):
    reference={}
    for name, path in shared.items():
        receipt=verify_directory(path,("continuous_reference_summary.json","private_c0.npz"))
        reference[name]=dict(summary=json.loads((path/"continuous_reference_summary.json").read_text()),
                             commit=json.loads(receipt.read_text())["commit"])
    cores={}
    strict={}
    deletions={}
    sampling={}
    for name,work in fits.items():
        verify_directory(work,("continuous_core_summary.json","private_qso_blocks.json","private_group_deletions.json",
                               "private_catalogue.csv.gz"))
        cores[name]=json.loads((work/"continuous_core_summary.json").read_text())
        strict[name]=json.loads((work/"private_qso_blocks.json").read_text())
        deletions[name]=json.loads((work/"private_group_deletions.json").read_text())
        sampling[name]=pd.read_csv(work/"private_catalogue.csv.gz",usecols=("r_bin",*SAMPLING_COLUMNS))
        if len(sampling[name])!=cores[name]["owned_w0"] or sampling[name].r_bin.isna().any():
            raise ValueError(f"W0 sampling records do not match the verified core {name}")
    gauges=defaultdict(list)
    for name, c in cores.items():
        gauges[(c["shared_solution_sha256"],c.get("reference_gauge","main"))].append(name)
    result={}
    for name, chosen in site_names.items():
        records=[row for site in chosen for row in strict[site]]
        grouped={}
        for low,hi in ((21,22),(22,23),(21,23)):
            selected=[v for v in records if low<=v["magnitude"]<hi]
            common=[v for v in selected if set(METHODS)<=v["motion"].keys()]
            grouped[f"{low}_to_{hi}"]=dict(strict=estimate(selected,("C0",),9137),
                                          common=estimate(common,METHODS,9137))
            if name not in fits:
                grouped[f"{low}_to_{hi}"]["common_by_sdss_runs"]=scan_strata(common)
        rows=[row for site in chosen for row in deletions[site]]
        samples=pd.concat([sampling[site] for site in chosen],ignore_index=True)
        result[name]=dict(bins=grouped,leave_group_out=deletes(rows),w0_sampling_by_r=summarize_sampling(samples))
    summaries=dict(
        total_core_area_sqdeg=sum(area(specs[k]) for k in cores),
        covered_cores={k:dict(ra_bounds_deg=specs[k]["ra"],dec_bounds_deg=specs[k]["dec"],area_sqdeg=area(specs[k]))
                       for k in cores}, cores=cores, cohorts=result,
        reference=reference, gauge_cores=list(gauges.values()),w0_sampling_columns=SAMPLING_COLUMNS)
    check_aggregate(summaries)
    return summaries


def geographic_sets(names, spec):
    sets={"all":list(names)}
    for name in sets["all"]:
        south,north=spec[name]["dec"]
        band="ordinary" if north<=-3 or south>=3 else "repeat_band" if -1<=south<=north<=1 else "transition"
        sets.setdefault(band,[]).append(name)
    return sets


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("output","plan","shared","config"):
        p.add_argument(name,type=Path)
    a=p.parse_args()
    files={k:Path(v) for k,v in json.loads(a.plan.read_text()).items()}
    graph_paths={k:Path(v) for k,v in json.loads(a.shared.read_text()).items()}
    spec=tomllib.loads(a.config.read_text())["fields"]
    sets={name:[name] for name in files}
    sets.update(geographic_sets(files,spec))
    save_json(a.output,pool(spec,files,graph_paths,sets))


if __name__=="__main__":
    main()
