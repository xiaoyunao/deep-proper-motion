"""Summarize strictly common QSO comparisons by observed NSC instrument."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .__main__ import tomllib
from .continuous_reference import verify_directory
from .continuous_summary import METHODS, estimate, geographic_sets
from .io import digest, save_json
from .public_summary import check_aggregate


def instrument_by_exposure(archives):
    instrument={}
    for p in archives:
        rows=pd.read_csv(p,usecols=["exposure","instrument"],dtype=str).dropna().drop_duplicates()
        for exp, value in rows.itertuples(index=False,name=None):
            name="NSC:"+exp
            if instrument.setdefault(name,value)!=value:
                raise ValueError("Conflicting instrument metadata for one physical exposure")
    return instrument


def instrument_bins(records, instrument):
    result={"21_to_22":defaultdict(list),"22_to_23":defaultdict(list)}
    for row in records:
        if not 21<=row["magnitude"]<23 or not set(METHODS)<=row["motion"].keys():
            continue
        keys={instrument.get(k,"unknown") for k in row["exposures"] if not k.startswith("SDSS:")}
        name="incomplete" if "unknown" in keys or not keys else "_and_".join(sorted(keys))
        result["21_to_22" if row["magnitude"]<22 else "22_to_23"][name].append(row)
    return result


def compare(fits, archives):
    instrument=instrument_by_exposure(archives)
    records=[row for core in fits for row in json.loads((core/"private_qso_blocks.json").read_text())]
    result={}
    for tag,groups in instrument_bins(records,instrument).items():
        summary={}
        for name, x in groups.items():
            summary[name]=dict(n=len(x))
            if len(x)>=5:
                v=np.asarray([[row["motion"][m] for m in METHODS] for row in x])
                summary[name].update(methods=METHODS,radial_p68_masyr=np.quantile(np.linalg.norm(v,axis=2),.68,axis=0))
        result[tag]=summary
    check_aggregate(result)
    return result


def compare_verified(fits, spec):
    archives=[]
    unavailable=[]
    records={}
    for name,core in fits.items():
        fit=verify_directory(core,("private_qso_blocks.json",))
        stage=json.loads(fit.read_text())["stages"]["continuous-core-v12"][-1]
        source=core/"source_receipt.json"
        if digest(source)!=stage["inputs"]["source_receipt.json"]:
            raise ValueError(f"Source receipt changed for {name}")
        source_dir=source.resolve().parent
        try:
            verify_directory(source_dir,("chips.csv",))
        except (ValueError,OSError):
            unavailable.append(name)
        else:
            archives.append(source_dir/"chips.csv")
        records[name]=json.loads((core/"private_qso_blocks.json").read_text())
    instrument=instrument_by_exposure(archives)
    cohorts={}
    for name, selected in geographic_sets(fits,spec).items():
        grouped=instrument_bins((r for k in selected for r in records[k]),instrument)
        cohorts[name]={tag:{k:estimate(rows,METHODS,9137) for k,rows in groups.items()}
                       for tag,groups in grouped.items()}
    result=dict(cores=len(fits),metadata_complete_cores=len(archives),
                missing_or_unverified_metadata_cores=unavailable,cohorts=cohorts)
    check_aggregate(result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("output",type=Path)
    p.add_argument("--fits",nargs="+",type=Path)
    p.add_argument("--chips",nargs="+",type=Path)
    p.add_argument("--plan",type=Path,help="Path to the checksum-verified fit directory mapping")
    p.add_argument("--config",type=Path)
    args=p.parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        p.error("Run on a Slurm compute node")
    if args.plan and args.config and not (args.fits or args.chips):
        fits={k:Path(v) for k,v in json.loads(args.plan.read_text()).items()}
        fields=tomllib.loads(args.config.read_text())["fields"]
        result=compare_verified(fits,fields)
    elif args.fits and args.chips and not (args.plan or args.config):
        result=compare(args.fits,args.chips)
    else:
        p.error("Supply --plan with --config, or --fits with --chips")
    save_json(args.output,result)


if __name__=="__main__":
    main()
