"""Summarize strictly common QSO comparisons by observed NSC instrument."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .continuous_summary import METHODS
from .io import save_json
from .public_summary import check_aggregate


def compare(fits, archives):
    instrument={}
    for p in archives:
        rows=pd.read_csv(p,usecols=["exposure","instrument"]).dropna().drop_duplicates()
        for exp, value in rows.itertuples(index=False,name=None):
            name="NSC:"+str(exp)
            if instrument.setdefault(name,value)!=value:
                raise ValueError("Conflicting instrument metadata for one physical exposure")
    result={}
    for tag, a, b in (("21_to_22",21,22),("22_to_23",22,23)):
        groups=defaultdict(list)
        for core in fits:
            for row in json.loads((core/"private_qso_blocks.json").read_text()):
                if not a<=row["magnitude"]<b or not set(METHODS)<=row["motion"].keys():
                    continue
                keys={instrument.get(k,"unknown") for k in row["exposures"] if not k.startswith("SDSS:")}
                name="incomplete" if "unknown" in keys or not keys else "_and_".join(sorted(keys))
                groups[name].append([row["motion"][m] for m in METHODS])
        summary={}
        for name, x in groups.items():
            summary[name]=dict(n=len(x))
            if len(x)>=5:
                v=np.asarray(x)
                summary[name].update(methods=METHODS,radial_p68_masyr=np.quantile(np.linalg.norm(v,axis=2),.68,axis=0))
        result[tag]=summary
    check_aggregate(result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("output",type=Path)
    p.add_argument("--fits",nargs="+",type=Path,required=True)
    p.add_argument("--chips",nargs="+",type=Path,required=True)
    args=p.parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        p.error("Run on a Slurm compute node")
    save_json(args.output,compare(args.fits,args.chips))


if __name__=="__main__":
    main()
