"""Publish spatial quality and observation-block uncertainty for completed cores."""
import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path

import numpy as np

from .io import digest, save_json
from .nsc_cohort import survey_overlap
from .nsc_comparison import comparison, configuration
from .provenance import run_stage
from .regional_validation import CONTRASTS, tag_magnitude


def load_output(directory,stage,filename):
    manifest=json.loads((directory/"run_manifest.json").read_text())
    receipt=manifest["stages"][stage][-1]
    path=directory/filename
    if receipt["state"]!="completed" or receipt["outputs"].get(filename)!=digest(path):
        raise ValueError(f"{directory.name}: invalid {stage} receipt")
    return json.loads(path.read_text())


def resampling(groups, settings, method, group_uncertainty=True, reference='published_nsc'):
    if not groups or (group_uncertainty and len(groups)<2):
        return None
    random=np.random.default_rng(settings["seed"])
    data=[np.array([(r[method][0],r[reference][0]) for r in group]) for group in groups]
    estimates=[]
    for _ in range(settings["bootstrap_draws"]):
        selected=[d[random.integers(len(d),size=len(d))]
                  for i in random.integers(len(data),size=len(data)) for d in (data[i],)]
        values=np.concatenate(selected)
        before,after=np.quantile(np.linalg.norm(values,axis=2),.68,axis=0)[::-1]
        estimates.append([*np.median(values[:,0,:],axis=0),1-after/before if before>0 else np.nan])
    draws=np.array(estimates)
    return dict(zero95_masyr=np.quantile(draws[:,:2],[.025,.975],axis=0).T,
                p68_gain95=np.nanquantile(draws[:,2],[.025,.975]) if np.isfinite(draws[:,2]).any() else None)


def field_area(spec):
    return 2*np.pi*(1-np.cos(np.deg2rad(spec["radius"])))*(180/np.pi)**2


def summarize(directories,settings,previous,observation_overlap=None):
    fields={}
    nested=defaultdict(lambda:defaultdict(dict))
    obs,overlap=(survey_overlap([*directories,*previous])
                 if observation_overlap is None else observation_overlap)
    valid_names={d.name for d in directories}
    independent=[sorted(set(b)&valid_names) for b in obs if set(b)&valid_names]
    options=settings["regional"]
    totals={method:defaultdict(Counter) for method in ("W0","W1","multi")}
    w0_exit=defaultdict(Counter)
    for d in directories:
        spec=settings["fields"][d.name]
        stage="regional-production-"+spec.get("series","v07")
        v=load_output(d,stage,"regional_summary.json")
        pairs=load_output(d,stage,"private_qso_pairs.json")
        for r,entry in pairs:
            for label in tag_magnitude(r,settings):
                for method in ("W0","W1","multi","same_three"):
                    required=("W0","W1","multi","published_nsc") if method=="same_three" else (method,"published_nsc")
                    if all(k in entry for k in required):
                        nested[label][method].setdefault(d.name,[]).append({k:entry[k] for k in required})
        local=nested["22_to_23"]["W0"].get(d.name,[])
        biopsy=v["qso"]["precision"].get("22_to_23",{}).get("W0",{})
        zero=resampling([local],settings,"W0",False) if len(local)>=settings["min_publish_count"] else None
        all_bracket=v["qso"]["precision"].get("21_to_23",{}).get("W0",{})
        faint=v["funnel"]["W0"].get("22_to_23",{})
        final=faint.get("final",0)
        initial=faint.get("initial",0)
        if not initial:
            fraction=None
        else:
            fraction=final/initial
        fail=[]
        if len(local)<options["quality_min_faint_qso"]:
            fail.append("insufficient_faint_qso")
        else:
            moments=biopsy.get("methods",{}).get("W0",{})
            if max(map(abs,moments["median"]))>options["quality_zero_limit"] or np.abs(zero["zero95_masyr"]).max()>options["quality_zero_limit"]:
                fail.append("faint_zero_point")
            gain=biopsy.get("paired",{}).get("W0_vs_published_nsc",{})
            delta=gain.get("fraction95",[None]*3)[2]
            if (delta is None or delta[0]<=0 or gain["fraction_improved"][2]<settings["practical_gain"]):
                fail.append("insufficient_faint_precision_gain")
        labels=v["gaia"]["association"]["W0"]["final_wrong_known_members"]
        if labels["wilson95"] is None or labels["wilson95"][1]>options["quality_max_known_member_contamination"]:
            fail.append("known_label_contamination_unconstrained")
        if fraction is None or fraction<options["quality_min_final_faint_fraction"]:
            fail.append("faint_yield")
        for method, counts in v["funnel"].items():
            for label,step in counts.items():
                totals[method][label].update(step)
        for label,exits in v["qso"]["strict_exits"].items():
            for reason,count in exits.items():
                w0_exit[label][reason]+=count["n"]
        fields[d.name]=dict(center_deg=[spec["ra"],spec["dec"]],role=spec["role"],area_sqdeg=field_area(spec),
                            area_overlap_basis="disjoint_spherical_cores",faint_w0=faint,zero95=zero,
                            faint_precision=biopsy,faint_21_to_23=all_bracket,quality_exits=fail,
                            known_label_contamination=labels,multiband=v["multiband"],
                            holdout=v["holdout"],calibration=v["calibration"],
                            repeating_sdss=v["coverage"].get("all",{}).get("repeated"),
                            acquisition=v["acquisition"],resources=v["resources"])
    pooled={}
    for label,methods in nested.items():
        for method,table in methods.items():
            rows=[r for seq in table.values() for r in seq]
            blocks=[[r for name in b for r in table.get(name,[])] for b in independent]
            blocks=[v for v in blocks if v]
            model="W0" if method=="same_three" else method
            pooled.setdefault(label,{})[method]=dict(fields=len(table),observation_blocks=len(blocks),
                 statistics=comparison(rows,settings,CONTRASTS),
                 block_bootstrap=resampling(blocks,settings,model))
            if method=="same_three":
                pooled[label][method]["multiband_vs_w1_block_bootstrap"]=resampling(
                    blocks,settings,"multi",reference="W1")
    passed=[name for name,v in fields.items() if not v["quality_exits"]]
    area=sum(v["area_sqdeg"] for v in fields.values())
    return dict(area_sqdeg=area,fields=fields,counts=totals,aggregate_qso_exits=w0_exit,
                pooled=pooled,observation_blocks=independent,
                shared_visits={k:v for k,v in overlap.items() if any(v.values())},
                quality_candidate_fields=passed,
                gate=dict(area_reached=area>=options["min_area_sqdeg"],
                          minimum_quality_region_count=2,
                          region_quality_reached=len(passed)>=2,
                          verified_external_dark_reference=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work",type=Path)
    parser.add_argument("--region",action="append",type=Path,required=True)
    parser.add_argument("--previous",nargs="*",type=Path,default=[])
    parser.add_argument("--config",type=Path,default=Path("configs/regional-v07.toml"))
    args=parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("Use a Slurm node.")
    args.work.mkdir(parents=True,exist_ok=True)
    tracked=[]
    for d in [*args.region,*args.previous]:
        names=["detections.csv"]
        if d in args.region:
            names.extend(["run_manifest.json","private_qso_pairs.json","regional_summary.json"])
        for f in names:
            path=args.work/d.name/f
            path.parent.mkdir(parents=True,exist_ok=True)
            target=(d/f).resolve(strict=True)
            if not path.exists():
                path.symlink_to(target)
            if path.resolve()!=target:
                parser.error("Use a new cohort directory for changed inputs.")
            tracked.append(str(path.relative_to(args.work)))
    out="regional_cohort_summary.json"
    with run_stage(args.work,args.config,"regional_pool","regional-pool-v07",inputs=tracked,outputs=[out]) as go:
        if go:
            _,settings=configuration(args.config)
            save_json(args.work/out,summarize(args.region,dict(settings,bootstrap_draws=1200),args.previous))


if __name__=="__main__":
    main()
