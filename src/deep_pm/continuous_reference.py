"""Prepare reference measurements per core and solve one common C0 graph."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import resource

import healpy as hp
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .__main__ import tomllib
from .astrometry import project
from .exposure_catalogues import inside
from .io import digest, save_json
from .pilot_production import quality_detections
from .precision_v03 import calibration_groups, reference_rows
from .provenance import run_stage
from .validation_audit import graph_support, quantiles
from .zero_point import fit_offsets

CENTER = (350., 0.)
PRECISION = Path("configs/precision-v03.toml")
SPATIAL = Path("configs/spatial-v08.toml")
FILES = ("detections.csv", "external_labels.csv", "spectra_veto.csv",
         "spectra_union.csv", "validation/gaia.csv", "official_nsc.csv", "acquisition_summary.json")


def verify_directory(path, files):
    receipt = path/"run_manifest.json"
    manifest = json.loads(receipt.read_text())
    recorded = {}
    for versions in manifest["stages"].values():
        stage = versions[-1]
        if stage["state"] == "completed":
            recorded.update(stage["outputs"])
    for name in files:
        if recorded.get(name) is None or digest(path/name) != recorded[name]:
            raise ValueError(f"Unverified input {path.name}/{name}")
    return receipt


def extract(source, work, spec):
    config = tomllib.loads(PRECISION.read_text())
    prod = tomllib.loads(SPATIAL.read_text())["production"]
    data, xy, time, sigma, keep = quality_detections(source,CENTER,config,prod)
    labels = pd.read_csv(source/"external_labels.csv",dtype=str).set_index("detection_id").original_object_id
    veto = pd.concat([pd.read_csv(source/name,usecols=["ra","dec"])
                      for name in ("spectra_veto.csv","validation/gaia.csv")],ignore_index=True).dropna()
    refs = reference_rows(data,xy,data.detection_id.map(labels),project(veto.ra,veto.dec,CENTER),
                          config["calibration"],config["seed"])
    i = refs.index.to_numpy()
    frame = refs[["reference","is_holdout","detection_id","exposure_id","ra_deg","dec_deg"]].copy()
    frame[["east","north"]] = xy[i]
    frame[["sigma_e","sigma_n"]] = sigma[i]
    frame["group"] = calibration_groups(data)[i]
    frame.to_csv(work/"reference.csv.gz",index=False)
    core = inside(data.ra_deg,data.dec_deg,spec)
    visits = data.loc[core&keep,["survey","exposure_id","observation_time"]].drop_duplicates()
    counts = dict(raw_core_valid_error=int(core.sum()), local_reference_rows=len(frame),
                  reference_sources=frame.reference.nunique(),
                  r_sdss_runs=visits.loc[visits.survey.eq("SDSS"),"exposure_id"].nunique(),
                  r_nsc_visits=visits.loc[visits.survey.eq("NSC"),"exposure_id"].nunique())
    save_json(work/"reference_summary.json", counts)


def deduplicate(frames):
    rows = pd.concat(frames,ignore_index=True)
    clashes = rows.groupby("detection_id").reference.transform("nunique").gt(1)
    cleaned = rows.loc[~clashes].drop_duplicates("detection_id").copy()
    # Archive processing defines a physical exposure, irrespective of how many cores retrieved it.
    duplicate_exposure = cleaned.duplicated(["reference","exposure_id"],keep=False)
    cleaned = cleaned.loc[~duplicate_exposure].reset_index(drop=True)
    if cleaned.groupby("reference").is_holdout.nunique().gt(1).any():
        raise ValueError("A physical reference source appears in both splits")
    return cleaned, dict(candidate_rows=len(rows), conflicting_match_rows=int(clashes.sum()),
                         multiple_exposure_rows=int(duplicate_exposure.sum()), unique_rows=len(cleaned))


def serialize_offsets(path, offsets, gauges=None):
    names = sorted(offsets)
    np.savez_compressed(path,names=np.asarray(names),
                        position=np.stack([offsets[k][0] for k in names]),
                        draws=np.stack([offsets[k][1] for k in names]),
                        counts=np.array([offsets[k][2] for k in names]),
                        gauge=np.asarray([gauges[k] if gauges else "main" for k in names]))


def read_offsets(path, with_gauge=False):
    with np.load(path, allow_pickle=False) as t:
        values = {k:(a,b,int(n)) for k,a,b,n in zip(t["names"],t["position"],t["draws"],t["counts"])}
        if with_gauge:
            identifiers = t["gauge"] if "gauge" in t else ["main"]*len(values)
            return values, dict(zip(values,identifiers))
    return values


def calibration_components(train, minimum):
    active=train
    while len(active):
        keep=active.loc[active.groupby("group").reference.transform("nunique")>=minimum]
        keep=keep.loc[keep.groupby("reference").group.transform("nunique")>=2]
        if len(keep)==len(active):
            break
        active=keep
    ir,ref=pd.factorize(active.reference)
    ig,group=pd.factorize(active.group)
    graph=coo_matrix((np.ones(len(ig),dtype=np.int8),(ir,len(ref)+ig)),
                     shape=(len(ref)+len(group),)*2)
    _,labels=connected_components(graph,directed=False)
    rows=active.assign(component=labels[ir])
    for _,subset in rows.groupby("component",sort=True):
        surveys={k.split(":")[0] for k in subset.group.unique()}
        if surveys=={"NSC","SDSS"}:
            yield subset


def held_reports(rows, offsets, specs, gauges):
    held = rows.loc[rows.is_holdout & rows.group.isin(offsets)].copy()
    if held.empty:
        return {}
    xy = held[["east","north"]].to_numpy()
    corrected = xy-np.stack([offsets[k][0] for k in held.group])
    for name, values in (("raw",xy),("fixed",corrected)):
        held[[name+"_e",name+"_n"]] = values
    holder = np.full(len(held),-1,dtype=int)
    for index, spec in enumerate(specs.values()):
        holder[inside(held.ra_deg,held.dec_deg,spec)] = index
    held["core_index"] = holder
    held["sdss"] = held.exposure_id.str.startswith("SDSS:")
    held["gauge"] = held.group.map(gauges)
    held["ring"] = hp.ang2pix(128,held.ra_deg,held.dec_deg,lonlat=True)
    from collections import defaultdict

    summaries = defaultdict(list)
    for _, group in held.groupby(["reference","gauge"],sort=False):
        c = group.core_index.to_numpy()
        xyz = group[["raw_e","raw_n","fixed_e","fixed_n"]].to_numpy().reshape(-1,2,2)
        side = group.sdss.to_numpy()
        if side.any() and (~side).any():
            summaries["cross_survey"].append(np.median(xyz[side],axis=0)-np.median(xyz[~side],axis=0))
        edges = group.ring.to_numpy()
        if len(set(edges))>1 and side.any() and (~side).any():
            summaries["cross_healpix"].append(np.median(xyz[side],axis=0)-np.median(xyz[~side],axis=0))
        cores = np.unique(c[c>=0])
        if len(cores)>1:
            for a,b in zip(cores[:-1],cores[1:]):
                summaries["cross_core"].append(np.median(xyz[c==a],axis=0)-np.median(xyz[c==b],axis=0))
    result = {name:dict(n=len(v),median_mas=np.median(v,axis=0),
                        radial_p68_mas=np.quantile(np.linalg.norm(v,axis=2),.68,axis=0))
              for name,v in summaries.items()}
    for key in ("cross_survey","cross_core","cross_healpix"):
        result.setdefault(key,{"n":0})
    return result


def solve(work, parts, specs, separate=False):
    datasets = {}
    for name, path in parts.items():
        verify_directory(path,("reference.csv.gz","reference_summary.json"))
        datasets[name] = pd.read_csv(path/"reference.csv.gz",dtype={"reference":str})
    rows, selection = deduplicate(list(datasets.values()))
    train = rows.loc[~rows.is_holdout]
    opts = tomllib.loads(PRECISION.read_text())
    cal = opts["calibration"]
    offsets, gauges = {}, {}
    pieces = calibration_components(train,cal["min_references"]) if separate else (train,)
    for component, part in enumerate(pieces):
        fit = fit_offsets(part.reference,part[["east","north"]].to_numpy(),
                          part[["sigma_e","sigma_n"]].to_numpy(),part.group.to_numpy(),
                          cal["min_references"],cal["bootstrap_draws"],opts["seed"]+component,
                          workers=int(os.environ.get("DEEP_PM_M3_WORKERS","1")))
        if not fit:
            continue
        late = [v for k,v in fit.items() if k.startswith("NSC:")]
        anchor = np.median(np.stack([v[0] for v in late]),axis=0)
        datum = np.median(np.stack([v[1]-v[1].mean(axis=0) for v in late]),axis=0)
        for k,(point,samples,n) in fit.items():
            centered = samples-samples.mean(axis=0)-datum
            if np.std(centered,axis=0,ddof=1).max() <= cal["max_offset_se_mas"]:
                offsets[k] = point-anchor,centered,n
                gauges[k] = str(component)
    if not offsets:
        raise ValueError("No jointly solvable SDSS–NSC component")
    serialize_offsets(work/"private_c0.npz",offsets,gauges)
    membership = Counter(k for d in datasets.values() for k in set(d.group) if k in offsets)
    by_survey = Counter(k.split(":")[0] for k in offsets)
    counts = {name:dict(reference_rows=len(d), safe_groups=len(set(d.group)&offsets.keys()),
                        unsupported_groups=len(set(d.group)-offsets.keys()))
              for name,d in datasets.items()}
    summary = dict(counts=selection, by_survey=by_survey, cores=counts, component_count=graph_support(
                       rows,rows.group.to_numpy(),offsets)["components"],
                   unsupported_groups=len(set(rows.group)-offsets.keys()),
                   shared_calibrated_groups=sum(v>1 for v in membership.values()),
                   shared_groups_in_adjacent={}, held=held_reports(rows,offsets,specs,gauges), applied_gauges=dict(Counter(gauges.values())),
                   offset_se_mas=quantiles([np.std(v[1],axis=0,ddof=1) for v in offsets.values()]),
                   max_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    for a, first in specs.items():
        for b, second in specs.items():
            if b<=a:
                continue
            adjacent = any(
                (first[edge][1]==second[edge][0] or second[edge][1]==first[edge][0])
                and min(first[along][1],second[along][1])>max(first[along][0],second[along][0])
                for edge, along in (("ra","dec"),("dec","ra")))
            if adjacent:
                aa,bb = datasets[a],datasets[b]
                shared = set(aa.group)&set(bb.group)&offsets.keys()
                held_a = aa.loc[aa.is_holdout,"reference"]
                held_b = bb.loc[bb.is_holdout,"reference"]
                summary["shared_groups_in_adjacent"][f"{a}__{b}"] = dict(
                    shared_groups=len(shared), common_reference_detections=len(set(aa.detection_id)&set(bb.detection_id)),
                    common_holdout_references=len(set(held_a)&set(held_b)))
    save_json(work/"continuous_reference_summary.json",summary)


def bind(work,name,target):
    link = work/name
    if not link.exists():
        link.symlink_to(target.resolve(strict=True))
    if link.resolve()!=target.resolve(strict=True):
        raise ValueError(f"Input changed for {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=["extract","solve"])
    parser.add_argument("work",type=Path)
    parser.add_argument("config",type=Path)
    parser.add_argument("region")
    parser.add_argument("source",type=Path)
    parser.add_argument("--separate-components",action="store_true")
    args = parser.parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        parser.error("Run on a Slurm compute node")
    args.work.mkdir(parents=True,exist_ok=True)
    specs = tomllib.loads(args.config.read_text())["fields"]
    if args.command=="extract":
        bind(args.work,"source_receipt.json",verify_directory(args.source,FILES))
        with run_stage(args.work,args.config,args.region,"continuous-ref-input-v12",
                       inputs=("source_receipt.json",),outputs=("reference.csv.gz","reference_summary.json")) as go:
            if go:
                extract(args.source,args.work,specs[args.region])
    else:
        parts = {k:Path(v) for k,v in json.loads(args.source.read_text()).items()}
        names = []
        for name, part in parts.items():
            alias = name+"_reference_receipt.json"
            bind(args.work,alias,part/"run_manifest.json")
            names.append(alias)
        with run_stage(args.work,args.config,args.region,"continuous-grid-v12",inputs=names,
                       outputs=("private_c0.npz","continuous_reference_summary.json")) as go:
            if go:
                solve(args.work,parts,{k:specs[k] for k in parts},args.separate_components)


if __name__ == "__main__":
    main()
