"""Respect shared SDSS runs and NSC exposures in dark-QSO comparisons."""
import argparse
import json
from pathlib import Path

import numpy as np


def connected(records, conservative_nights=False):
    parents=list(range(len(records)))
    def root(i):
        while parents[i]!=i:
            parents[i]=parents[parents[i]]
            i=parents[i]
        return i
    owners={}
    for i,record in enumerate(records):
        groups=record["exposures"]
        if conservative_nights:
            groups=groups+[f"NSC-night:{j}" for j in record["nsc_epoch_days"]]
        for group in groups:
            parents[root(i)]=root(owners.setdefault(group,i))
    result={}
    for i in range(len(records)):
        result.setdefault(root(i),[]).append(i)
    return list(result.values())


def metrics(records, conservative_nights=False):
    components=connected(records,conservative_nights)
    result=dict(n=len(records), block_sizes=sorted(map(len,components),reverse=True), independent_blocks=len(components))
    if not records:
        return result
    vectors=np.array([[r["motion"][k] for k in ("C0","C1")] for r in records])
    norms=np.linalg.norm(vectors,axis=2)
    values=np.quantile(norms,.68,axis=0)
    result.update(radial_p68_masyr=values.tolist(), fractional_gain=float(1-values[1]/values[0]),
                  axes_zero_masyr=np.median(vectors,axis=0).tolist())
    if len(components)>=3:
        random=np.random.default_rng(9137)
        boot=[]
        zero=[]
        for selected in random.integers(len(components),size=(2000,len(components))):
            i=np.concatenate([components[j] for j in selected])
            q=np.quantile(norms[i],.68,axis=0)
            if q[0]>0:
                boot.append(1-q[1]/q[0])
            zero.append(np.median(vectors[i],axis=0))
        result.update(fractional_gain95=np.quantile(boot,[.025,.975]).tolist(),
                      axes_zero95=np.quantile(zero,[.025,.975],axis=0).transpose(1,2,0).tolist())
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("root",type=Path)
    p.add_argument("sites",nargs="+")
    args=p.parse_args()
    data={name:json.loads((args.root/name/"qso_observation_blocks.json").read_text()) for name in args.sites}
    combined=[r for entry in data.values() for r in entry]
    output={}
    for tag,mask in (("21_to_23",lambda r:True),("22_to_23",lambda r:r["magnitude"]>=22)):
        filtered={k:[r for r in v if mask(r)] for k,v in dict(combined=combined,**data).items()}
        output[tag]={k:dict(exposure=metrics(v),night=metrics(v,True)) for k,v in filtered.items()}
    print(json.dumps(output,allow_nan=False))


if __name__=="__main__":
    main()
