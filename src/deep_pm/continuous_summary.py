"""Summarize C0 trial coverage, common catalogues and visit-level dependence."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

from .__main__ import tomllib
from .continuous_reference import verify_directory
from .io import save_json
from .public_summary import check_aggregate
from .validation_blocks import connected

METHODS=("NSC","Xiao","C0")


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


def resample(records, vectors, seed):
    random=np.random.default_rng(seed)
    result={}
    for kind in ("sdss_run","nsc_exposure","nsc_night"):
        matrix=membership(records,kind)
        n=len(records)
        groups=matrix.shape[1]
        raw_load=np.asarray(matrix.sum(axis=0)).ravel()
        info=(raw_load.sum()**2/(raw_load@raw_load)) if groups else 0
        output=dict(groups=groups,equal_weight_visit_information=float(info))
        if groups>=3 and n>=5:
            weights=matrix @ random.exponential(size=(groups,800))
            order=np.argsort(vectors,axis=0)
            median=[]
            for j in range(vectors.shape[1]):
                selected=order[:,j]
                cum=weights[selected].cumsum(axis=0)
                k=(cum >= cum[-1:]*.5).argmax(axis=0)
                median.append(vectors[selected[k],j])
            output["weighted_zero95"]=np.quantile(median,[.025,.975],axis=1).T
        result[kind]=output
    return result


def estimate(records, methods, seed):
    if len(records)<5:
        return {"n":len(records)}
    vectors=np.array([[v["motion"][m] for m in methods] for v in records])
    results={}
    for k,m in enumerate(methods):
        value=vectors[:,k]
        median=np.median(value,axis=0)
        norm=np.linalg.norm(value,axis=1)
        results[m]=dict(median=median,nmad=1.4826*np.median(abs(value-median),axis=0),
                        radial_p68=np.quantile(norm,.68),tail_gt_5=int(np.count_nonzero(norm>5)),
                        visit_weighted_zero=resample(records,value,seed))
    return dict(n=len(records),methods=results,connected_sizes=sorted(map(len,connected(records)),reverse=True),
                shared_night_sizes=sorted(map(len,connected(records,True)),reverse=True))


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


def pool(specs, fits, shared, site_names):
    reference={}
    for name, path in shared.items():
        receipt=verify_directory(path,("continuous_reference_summary.json","private_c0.npz"))
        reference[name]=dict(summary=json.loads((path/"continuous_reference_summary.json").read_text()),
                             commit=json.loads(receipt.read_text())["commit"])
    cores={}
    strict={}
    deletions={}
    for name,work in fits.items():
        verify_directory(work,("continuous_core_summary.json","private_qso_blocks.json","private_group_deletions.json"))
        cores[name]=json.loads((work/"continuous_core_summary.json").read_text())
        strict[name]=json.loads((work/"private_qso_blocks.json").read_text())
        deletions[name]=json.loads((work/"private_group_deletions.json").read_text())
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
        rows=[row for site in chosen for row in deletions[site]]
        result[name]=dict(bins=grouped,leave_group_out=deletes(rows))
    summaries=dict(
        total_core_area_sqdeg=sum(area(specs[k]) for k in cores),
        covered_cores={k:dict(ra_bounds_deg=specs[k]["ra"],dec_bounds_deg=specs[k]["dec"],area_sqdeg=area(specs[k]))
                       for k in cores}, cores=cores, cohorts=result,
        reference=reference, gauge_cores=list(gauges.values()))
    check_aggregate(summaries)
    return summaries


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("output","plan","shared","config"):
        p.add_argument(name,type=Path)
    a=p.parse_args()
    files={k:Path(v) for k,v in json.loads(a.plan.read_text()).items()}
    graph_paths={k:Path(v) for k,v in json.loads(a.shared.read_text()).items()}
    sets={"all":list(files)}
    ordinary=[name for name in files if name!="repeat_rich"]
    if ordinary and ordinary!=list(files):
        sets["ordinary"]=ordinary
    for name in files:
        sets[name]=[name]
    spec=tomllib.loads(a.config.read_text())["fields"]
    save_json(a.output,pool(spec,files,graph_paths,sets))


if __name__=="__main__":
    main()
