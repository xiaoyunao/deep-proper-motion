"""Prepare reference measurements per core and solve one common C0 graph."""
import argparse
from collections import Counter
from functools import lru_cache
from hashlib import sha256
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


def extract(source, work, spec, center=CENTER):
    config = tomllib.loads(PRECISION.read_text())
    prod = tomllib.loads(SPATIAL.read_text())["production"]
    data, xy, time, sigma, keep = quality_detections(source,center,config,prod)
    labels = pd.read_csv(source/"external_labels.csv",dtype=str).set_index("detection_id").original_object_id
    veto = pd.concat([pd.read_csv(source/name,usecols=["ra","dec"])
                      for name in ("spectra_veto.csv","validation/gaia.csv")],ignore_index=True).dropna()
    refs = reference_rows(data,xy,data.detection_id.map(labels),project(veto.ra,veto.dec,center),
                          config["calibration"],config["seed"])
    i = refs.index.to_numpy()
    frame = refs[["reference","is_holdout","detection_id","exposure_id","ra_deg","dec_deg"]].copy()
    frame[["east","north"]] = xy[i]
    frame[["sigma_e","sigma_n"]] = sigma[i]
    frame["group"] = calibration_groups(data)[i]
    frame.to_csv(work/"reference.csv.gz",index=False)
    core = inside(data.ra_deg,data.dec_deg,spec)
    visits = data.loc[core&keep,["survey","exposure_id","observation_time"]].drop_duplicates()
    counts = dict(projection_center_deg=list(center), raw_core_valid_error=int(core.sum()), local_reference_rows=len(frame),
                  reference_sources=frame.reference.nunique(),
                  r_sdss_runs=visits.loc[visits.survey.eq("SDSS"),"exposure_id"].nunique(),
                  r_nsc_visits=visits.loc[visits.survey.eq("NSC"),"exposure_id"].nunique())
    save_json(work/"reference_summary.json", counts)


def deduplicate(frames):
    rows = pd.concat(frames,ignore_index=True)
    if not set(pd.unique(rows.is_holdout)) <= {True,False}:
        raise ValueError('Reference split labels must be explicit booleans')
    rows['is_holdout']=rows.is_holdout.astype(bool)
    clashes = rows.groupby("detection_id").reference.transform("nunique").gt(1)
    cleaned = rows.loc[~clashes].drop_duplicates("detection_id").copy()
    # Archive processing defines a physical exposure, irrespective of how many cores retrieved it.
    duplicate_exposure = cleaned.duplicated(["reference","exposure_id"],keep=False)
    cleaned = cleaned.loc[~duplicate_exposure].reset_index(drop=True)
    if cleaned.groupby("reference").is_holdout.nunique().gt(1).any():
        raise ValueError("A physical reference source appears in both splits")
    return cleaned, dict(candidate_rows=len(rows), conflicting_match_rows=int(clashes.sum()),
                         multiple_exposure_rows=int(duplicate_exposure.sum()), unique_rows=len(cleaned))


def serialize_offsets(path, offsets, gauges=None, center=CENTER):
    names = sorted(offsets)
    np.savez_compressed(path,names=np.asarray(names),
                        position=np.stack([offsets[k][0] for k in names]),
                        draws=np.stack([offsets[k][1] for k in names]),
                        counts=np.array([offsets[k][2] for k in names]),
                        gauge=np.asarray([gauges[k] if gauges else "main" for k in names]),
                        projection_center_deg=np.asarray(center,dtype=float))


def reference_frame(path):
    """Old v1.2 snapshots use CENTER; newer graphs bind the actual projection."""
    from .stereographic import from_record
    with np.load(path,allow_pickle=False) as t:
        return from_record(t) if 'projection_center_deg' in t else CENTER


def common_reference_frame(metadata):
    centers = [tuple(m.get('projection_center_deg',CENTER)) for m in metadata]
    if not centers or len(set(centers)) != 1:
        raise ValueError('Reference shards must share one explicitly known projection')
    return centers[0]


def read_offsets(path, with_gauge=False):
    with np.load(path, allow_pickle=False) as t:
        if 'feature_coefficients' in t:
            raise ValueError('Chromatic reference requires per-source color and synchronous feature draws')
        values = {k:(a,b,int(n)) for k,a,b,n in zip(t["names"],t["position"],t["draws"],t["counts"])}
        if with_gauge:
            identifiers = t["gauge"] if "gauge" in t else ["main"]*len(values)
            return values, dict(zip(values,identifiers))
    return values


def calibration_components(train, minimum, trace=None):
    active=train
    while len(active):
        keep=active.loc[active.groupby("group").reference.transform("nunique")>=minimum]
        keep=keep.loc[keep.groupby("reference").group.transform("nunique")>=2]
        if len(keep)==len(active):
            break
        active=keep
    if trace is not None:
        trace['peeled_group_references']=active.groupby('group').reference.nunique().to_dict()
    ir,ref=pd.factorize(active.reference)
    ig,group=pd.factorize(active.group)
    graph=coo_matrix((np.ones(len(ig),dtype=np.int8),(ir,len(ref)+ig)),
                     shape=(len(ref)+len(group),)*2)
    _,labels=connected_components(graph,directed=False)
    rows=active.assign(component=labels[ir])
    if trace is not None:
        trace['peeled_components']={str(k):dict(groups=part.group.nunique(),references=part.reference.nunique(),
            surveys=sorted({v.split(':')[0] for v in part.group.unique()})) for k,part in rows.groupby('component',sort=True)}
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


def solve(work, parts, specs, separate=False, diagnostic_groups=False, checkpoint_dir=None):
    if diagnostic_groups and not separate:
        raise ValueError('Group-selection tracing requires explicit separate components')
    datasets = {}
    metadata = []
    receipt_checksums = {}
    for name, path in parts.items():
        receipt = verify_directory(path, ("reference.csv.gz", "reference_summary.json"))
        receipt_checksums[name] = digest(receipt)
        datasets[name] = pd.read_csv(path/"reference.csv.gz",dtype={"reference":str,"is_holdout":bool})
        metadata.append(json.loads((path/'reference_summary.json').read_text()))
    center = common_reference_frame(metadata)
    rows, selection = deduplicate(list(datasets.values()))
    # Release the original halo-overlapping frames before building the fit matrices.
    del datasets
    graph_rows = rows[["reference", "is_holdout", "group"]].copy()
    held_columns = ["reference", "is_holdout", "group", "ra_deg", "dec_deg", "east", "north", "exposure_id"]
    held_rows = rows.loc[rows.is_holdout, held_columns].copy()
    train = rows.loc[~rows.is_holdout, ["reference", "group", "east", "north", "sigma_e", "sigma_n"]].copy()
    del rows

    @lru_cache(maxsize=8)
    def audit_shard(name):
        path = parts[name]
        receipt = verify_directory(path, ("reference.csv.gz",))
        if digest(receipt) != receipt_checksums[name]:
            raise ValueError("The reference shard changed during the solve")
        return pd.read_csv(path/"reference.csv.gz",
                           usecols=["reference", "group", "detection_id", "is_holdout"],
                           dtype={"reference": str, "is_holdout": bool})
    opts = tomllib.loads(PRECISION.read_text())
    cal = opts["calibration"]
    checkpoint = None
    if checkpoint_dir is not None:
        manifest = json.loads((work/"run_manifest.json").read_text())
        binding = dict(commit=manifest["commit"], config_sha256=manifest["config_sha256"],
                       receipts=list(receipt_checksums.items()), calibration=cal,
                       seed=opts["seed"], separate_components=separate)
        checkpoint = checkpoint_dir/sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
    offsets, gauges = {}, {}
    trace={} if diagnostic_groups else None
    pieces = calibration_components(train,cal["min_references"],trace) if separate else (train,)
    unfiltered,unfiltered_gauges,diagnostics={},{},{}
    for component, part in enumerate(pieces):
        fit = fit_offsets(part.reference,part[["east","north"]].to_numpy(),
                          part[["sigma_e","sigma_n"]].to_numpy(),part.group.to_numpy(),
                          cal["min_references"],cal["bootstrap_draws"],opts["seed"]+component,
                          workers=int(os.environ.get("DEEP_PM_M3_WORKERS","1")),
                          checkpoint=checkpoint/f"component-{component:04d}" if checkpoint is not None else None)
        if not fit:
            continue
        late = [v for k,v in fit.items() if k.startswith("NSC:")]
        anchor = np.median(np.stack([v[0] for v in late]),axis=0)
        datum = np.median(np.stack([v[1]-v[1].mean(axis=0) for v in late]),axis=0)
        for k,(point,samples,n) in fit.items():
            centered = samples-samples.mean(axis=0)-datum
            error=np.std(centered,axis=0,ddof=1)
            accepted=error.max() <= cal["max_offset_se_mas"]
            if diagnostic_groups:
                unfiltered[k]=point-anchor,centered,n
                unfiltered_gauges[k]=str(component)
                diagnostics[k]=dict(fit_component=str(component),fit_reference_count=n,se_e_mas=error[0],se_n_mas=error[1],
                    status='retained' if accepted else 'offset_se_rejected')
            if accepted:
                offsets[k] = point-anchor,centered,n
                gauges[k] = str(component)
    if not offsets:
        raise ValueError("No jointly solvable SDSS–NSC component")
    diagnostic_summary=None
    if diagnostic_groups:
        group_names=sorted(set(graph_rows.group))
        raw_counts=graph_rows.groupby('group').reference.nunique()
        training_counts=train.groupby('group').reference.nunique()
        records=[]
        peeled=trace.get('peeled_group_references',{}) if trace is not None else {}
        for key in group_names:
            base=dict(group=key,raw_reference_count=int(raw_counts.get(key,0)),
                training_reference_count=int(training_counts.get(key,0)),peeled_reference_count=int(peeled.get(key,0)))
            reason='peeled_or_no_joint_component'
            if training_counts.get(key,0)<cal['min_references']:reason='below_initial_training_reference_minimum'
            records.append(dict(base,**diagnostics.get(key,dict(status=reason))))
        table=pd.DataFrame(records);table.to_csv(work/'private_reference_group_selection.csv.gz',index=False)
        serialize_offsets(work/'private_c0_unfiltered.npz',unfiltered,unfiltered_gauges,center)
        diagnostic_summary=dict(groups_before_selection=len(group_names),training_groups=int(len(training_counts)),
            groups_after_peeling=len(peeled),groups_before_offset_se_cut=len(unfiltered),
            selection_counts=table.status.value_counts().to_dict(),peeled_components=trace.get('peeled_components',{}),
            se_by_status={status:dict(group_count=len(part),p16_p50_p84_mas=quantiles(part[['se_e_mas','se_n_mas']].to_numpy()) if len(part)>=5 else None)
                for status,part in table.loc[table.se_e_mas.notna()].groupby('status')},
            core_selection={name:dict(Counter(table.set_index('group').status.reindex(sorted(set(audit_shard(name).group))).dropna())) for name in parts},
            scope='Trace only. Original support, gauge, weights, bootstrap and 12mas SE selection unchanged.')
    serialize_offsets(work/"private_c0.npz",offsets,gauges,center)
    del train, pieces, part
    membership = Counter()
    counts = {}
    for name in parts:
        shard = audit_shard(name)
        groups = set(shard.group)
        membership.update(groups & offsets.keys())
        counts[name] = dict(reference_rows=len(shard), safe_groups=len(groups & offsets.keys()),
                            unsupported_groups=len(groups - offsets.keys()))
    by_survey = Counter(k.split(":")[0] for k in offsets)
    summary = dict(projection_center_deg=center, counts=selection, by_survey=by_survey, cores=counts, component_count=graph_support(
                       graph_rows,graph_rows.group.to_numpy(),offsets)["components"],
                   unsupported_groups=len(set(graph_rows.group)-offsets.keys()),
                   shared_calibrated_groups=sum(v>1 for v in membership.values()),
                   shared_groups_in_adjacent={}, held=held_reports(held_rows,offsets,specs,gauges), applied_gauges=dict(Counter(gauges.values())),
                   offset_se_mas=quantiles([np.std(v[1],axis=0,ddof=1) for v in offsets.values()]),
                   max_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    if diagnostic_summary is not None:summary['diagnostic_group_selection']=diagnostic_summary
    for a, first in specs.items():
        for b, second in specs.items():
            if b<=a:
                continue
            adjacent = any(
                (first[edge][1]==second[edge][0] or second[edge][1]==first[edge][0])
                and min(first[along][1],second[along][1])>max(first[along][0],second[along][0])
                for edge, along in (("ra","dec"),("dec","ra")))
            if adjacent:
                aa,bb = audit_shard(a),audit_shard(b)
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
    parser.add_argument('--diagnostic-groups',action='store_true',help='Save private pre-SE offsets and group-selection reasons without changing C0')
    parser.add_argument("--checkpoint-dir", type=Path, help="Save verified intermediate reference fits and bootstrap draws")
    args = parser.parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        parser.error("Run on a Slurm compute node")
    args.work.mkdir(parents=True,exist_ok=True)
    run_config = tomllib.loads(args.config.read_text())
    specs = run_config["fields"]
    if args.command=="extract":
        bind(args.work,"source_receipt.json",verify_directory(args.source,FILES))
        with run_stage(args.work,args.config,args.region,"continuous-ref-input-v12",
                       inputs=("source_receipt.json",),outputs=("reference.csv.gz","reference_summary.json")) as go:
            if go:
                extract(args.source,args.work,specs[args.region],tuple(run_config.get('projection_center_deg',CENTER)))
    else:
        parts = {k:Path(v) for k,v in json.loads(args.source.read_text()).items()}
        names = []
        for name, part in parts.items():
            alias = name+"_reference_receipt.json"
            bind(args.work,alias,part/"run_manifest.json")
            names.append(alias)
        outputs=("private_c0.npz","continuous_reference_summary.json")
        if args.diagnostic_groups:
            option=args.work/'diagnostic_options.json'
            if option.exists() and json.loads(option.read_text())!={'diagnostic_groups':True}:
                parser.error('Diagnostic options changed; use a new versioned directory')
            if not option.exists():save_json(option,dict(diagnostic_groups=True))
            names.append('diagnostic_options.json')
            outputs+=('private_c0_unfiltered.npz','private_reference_group_selection.csv.gz')
        with run_stage(args.work,args.config,args.region,"continuous-grid-v12",inputs=names,
                       outputs=outputs) as go:
            if go:
                solve(args.work,parts,{k:specs[k] for k in parts},args.separate_components,args.diagnostic_groups,
                      args.checkpoint_dir)


if __name__ == "__main__":
    main()
