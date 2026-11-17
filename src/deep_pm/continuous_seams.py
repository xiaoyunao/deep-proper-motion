"""Compare independent held galaxy residuals immediately on either side of a seam."""
import argparse
from itertools import combinations
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .__main__ import tomllib
from .continuous_reference import deduplicate, read_offsets, verify_directory
from .io import digest, save_json
from .public_summary import check_aggregate


def held_by_reference(parts, solution, gauges):
    rows, _ = deduplicate(parts)
    rows = rows.loc[rows.is_holdout & rows.group.isin(solution)].copy()
    xy = rows[["east","north"]].to_numpy()
    fixed = xy-np.stack([solution[k][0] for k in rows.group]) if len(rows) else xy
    rows[["fixed_e","fixed_n"]] = fixed
    rows["sdss"] = rows.group.str.startswith("SDSS:")
    rows["gauge"] = rows.group.map(gauges) if gauges is not None else "main"
    records = []
    for (ref,gauge), g in rows.groupby(["reference","gauge"],sort=False):
        early=g.sdss.to_numpy()
        if early.any() and (~early).any():
            points=g[["east","north","fixed_e","fixed_n"]].to_numpy().reshape(-1,2,2)
            sky=g[["ra_deg","dec_deg"]].median().to_numpy()
            residual=(np.median(points[early],axis=0)-np.median(points[~early],axis=0)).ravel()
            records.append([ref,gauge,len(g),*sky,*residual])
    cols=["reference","gauge","n","ra","dec","raw_e","raw_n","fixed_e","fixed_n"]
    table=pd.DataFrame(records,columns=cols)
    # If separate graph components both constrain a galaxy, keep the one
    # with the most detections. One object may appear on only one side.
    return table.sort_values(["n","gauge"],ascending=[False,True]).drop_duplicates("reference")


def seam_segments(specs):
    for (name,a),(other,b) in combinations(specs.items(),2):
        for coord,cross in (("ra","dec"),("dec","ra")):
            lo,hi=max(a[cross][0],b[cross][0]),min(a[cross][1],b[cross][1])
            if lo>=hi:
                continue
            edges={x for x,y in ((a[coord][1],b[coord][0]),(b[coord][1],a[coord][0])) if x==y}
            if len(edges)==1:
                yield f"{name}__{other}",coord,cross,edges.pop(),lo,hi


def summarize_strips(negative, positive):
    columns=["raw_e","raw_n","fixed_e","fixed_n"]
    portions=[v[columns].to_numpy().reshape(-1,2,2) for v in (negative,positive)]
    stats=dict(held_independent_refs=[len(v) for v in portions])
    if all(len(v)>=5 for v in portions):
        median=[np.median(v,axis=0) for v in portions]
        stats.update(median_positive_minus_negative_mas=median[1]-median[0],
                     radial_p68_mas=[np.quantile(np.linalg.norm(v,axis=2),.68,axis=0) for v in portions])
    return stats


def adjacent_held(parts, specs, solution, width=2/60, *, gauges=None):
    table=held_by_reference(parts,solution,gauges)
    result={}
    for name,coord,cross,line,lo,hi in seam_segments(specs):
        offset=table[coord]-line
        whole=table.loc[(offset.abs()<width)&table[cross].between(lo,hi,inclusive="left")]
        negative=whole.loc[whole[coord]<line]
        positive=whole.loc[whole[coord]>=line]
        stats=dict(axis=coord,strip_width_arcmin=width*60)
        if gauges is None:
            stats.update(summarize_strips(negative,positive))
        else:
            stats.update(boundary_deg=line,edge_extent_deg=[lo,hi],
                         held_independent_refs=[len(negative),len(positive)])
            stats["by_gauge"]={str(g):summarize_strips(negative.loc[negative.gauge==g],positive.loc[positive.gauge==g])
                               for g in sorted(set(whole.gauge))}
        result[name]=stats
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("output",type=Path)
    p.add_argument("shared",type=Path)
    p.add_argument("refs",type=Path)
    p.add_argument("config",type=Path)
    p.add_argument("fields",nargs="*")
    p.add_argument("--plan",action="store_true",help="Read the reference input mapping for a shared graph")
    a=p.parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        p.error("Run on a Slurm compute node")
    graph_receipt=verify_directory(a.shared,("private_c0.npz",))
    fields=tomllib.loads(a.config.read_text())["fields"]
    if a.plan and not a.fields:
        refs={k:Path(v) for k,v in json.loads(a.refs.read_text()).items()}
        graph=json.loads(graph_receipt.read_text())["stages"]["continuous-grid-v12"][-1]
        supplied={name+"_reference_receipt.json":digest(p/"run_manifest.json") for name,p in refs.items()}
        if supplied!=graph["inputs"]:
            raise ValueError("The seam reference plan differs from the solved graph")
    elif a.fields and not a.plan:
        refs={k:a.refs/k for k in a.fields}
    else:
        p.error("Supply named fields or --plan with the reference input mapping")
    parts=[]
    for ref in refs.values():
        verify_directory(ref,("reference.csv.gz",))
        parts.append(pd.read_csv(ref/"reference.csv.gz",dtype={"reference":str}))
    offsets=a.shared/"private_c0.npz"
    specs={k:fields[k] for k in refs}
    if a.plan:
        solution,gauges=read_offsets(offsets,with_gauge=True)
        seams=adjacent_held(parts,specs,solution,gauges=gauges)
        result=dict(reference_graph_sha256=digest(offsets),cores=len(refs),edges=seams)
    else:
        result=adjacent_held(parts,specs,read_offsets(offsets))
    check_aggregate(result)
    save_json(a.output,result)


if __name__=="__main__":
    main()
