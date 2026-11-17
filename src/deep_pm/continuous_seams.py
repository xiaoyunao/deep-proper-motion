"""Compare independent held galaxy residuals immediately on either side of a seam."""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .__main__ import tomllib
from .continuous_reference import deduplicate, read_offsets, verify_directory
from .io import save_json


def adjacent_held(parts, specs, solution, width=2/60):
    rows, _ = deduplicate(parts)
    rows = rows.loc[rows.is_holdout & rows.group.isin(solution)].copy()
    xy = rows[["east","north"]].to_numpy()
    fixed = xy-np.stack([solution[k][0] for k in rows.group]) if len(rows) else xy
    rows[["fixed_e","fixed_n"]] = fixed
    rows["sdss"] = rows.group.str.startswith("SDSS:")
    ensemble = []
    for _, g in rows.groupby("reference",sort=False):
        early=g.sdss.to_numpy()
        if early.any() and (~early).any():
            points=g[["east","north","fixed_e","fixed_n"]].to_numpy().reshape(-1,2,2)
            sky=g[["ra_deg","dec_deg"]].median().to_numpy()
            ensemble.append([*sky,*(np.median(points[early],axis=0)-np.median(points[~early],axis=0)).ravel()])
    values=np.asarray(ensemble).reshape(-1,6)
    result={}
    for i,(name,a) in enumerate(specs.items()):
        for other,b in list(specs.items())[i+1:]:
            for axis, coord in enumerate(("ra","dec")):
                cross=("dec","ra")[axis]
                lo, hi = max(a[cross][0],b[cross][0]), min(a[cross][1],b[cross][1])
                if lo>=hi:
                    continue
                edges={x for x,y in ((a[coord][1],b[coord][0]),(b[coord][1],a[coord][0])) if x==y}
                if len(edges)!=1:
                    continue
                line=edges.pop()
                offset=values[:,axis]-line
                across=values[:,1-axis]
                window=(np.abs(offset)<width)&(across>=lo)&(across<hi)
                portions=[values[window&side,2:].reshape(-1,2,2) for side in (offset<0,offset>=0)]
                stats=dict(axis=coord,held_independent_refs=[len(v) for v in portions],strip_width_arcmin=width*60)
                if all(len(v)>=5 for v in portions):
                    median=[np.median(v,axis=0) for v in portions]
                    stats.update(median_positive_minus_negative_mas=median[1]-median[0],
                                 radial_p68_mas=[np.quantile(np.linalg.norm(v,axis=2),.68,axis=0) for v in portions])
                result[f"{name}__{other}"]=stats
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("output",type=Path)
    p.add_argument("shared",type=Path)
    p.add_argument("refs",type=Path)
    p.add_argument("config",type=Path)
    p.add_argument("fields",nargs="+")
    a=p.parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        p.error("Run on a Slurm compute node")
    verify_directory(a.shared,("private_c0.npz",))
    fields=tomllib.loads(a.config.read_text())["fields"]
    parts=[]
    for k in a.fields:
        verify_directory(a.refs/k,("reference.csv.gz",))
        parts.append(pd.read_csv(a.refs/k/"reference.csv.gz",dtype={"reference":str}))
    save_json(a.output,adjacent_held(parts,{k:fields[k] for k in a.fields},read_offsets(a.shared/"private_c0.npz")))


if __name__=="__main__":
    main()
