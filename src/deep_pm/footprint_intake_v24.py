"""Run registered all-filter acquisition and spherical reference extraction on Slurm."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import json
import os
from pathlib import Path
import resource
import subprocess
from time import monotonic

from .continuous_summary import area
from .exposure_inputs import acquire
from .full_reference_input import extract_one
from .io import save_json,digest
from .provenance import run_stage

OUTPUTS=("detections.csv","external_labels.csv","official_nsc.csv","spectra_union.csv","spectra_veto.csv",
         "validation/gaia.csv","acquisition_summary.json","sdss.csv","nsc_object.csv","chips.csv")
SETTINGS=dict(halo_deg=.06,native_join=True,native_count_window=True,native_row_cap=80000,nsc_all_objects=True,download_workers=1)


def select_pilots(cells):
    # Sky quadrant strata, before source/PM files are read. Order by stable hash.
    populations=[[] for _ in range(4)]
    for label,spec in cells.items():
        a=sum(spec["ra"])/2;d=sum(spec["dec"])/2
        if area(spec)<.2 or spec["time_area"] < .6*area(spec):continue
        group=2*int(d>=15)+int(a>=180)
        populations[group].append(label)
    return [min(group,key=lambda name:sha256(("v24-intake:"+name).encode()).digest()) for group in populations if group]


def one(task, revision=None):
    label,spec,base,catalogues,sas,manifest_sha=task
    root=base/"v2.4"
    original=root/"intake"/label
    prior=original/"run_manifest.json"
    if revision is None and prior.is_file():
        previous=json.loads(prior.read_text())
        if previous["commit"]!=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip():
            revision="canonical-longitude"
    work=root/"revisions"/revision if revision is not None else root
    dest=work/"intake"/label
    dest.mkdir(exist_ok=True,parents=True)
    if revision and (original/"cache").is_dir() and not (dest/"cache").exists():
        (dest/"cache").symlink_to((original/"cache").resolve(),target_is_directory=True)
    config=dest/"registered.toml"
    text=(f'manifest_sha256 = "{manifest_sha}"\n'+
          '\n'.join(f'{k} = {str(v).lower()}' for k,v in SETTINGS.items())+
          f'\n[fields.{label}]\nra = {spec["ra"]}\ndec = {spec["dec"]}\n')
    if config.exists() and config.read_text()!=text:raise ValueError("Changed acquisition configuration")
    config.write_text(text)
    t=monotonic()
    with run_stage(dest,config,label,"v24-all-measurements",outputs=OUTPUTS) as go:
        if go:
            acquire(dest,spec,SETTINGS,catalogues,sas,1)
            if not all((dest/name).is_file() for name in OUTPUTS):raise ValueError("Missing acquisition output")
    ref=extract_one((dict(core=label,source=str(dest),source_receipt_sha256=digest(dest/"run_manifest.json"),
                          center=[sum(spec[k])/2 for k in ("ra","dec")]),work/"input"))
    if revision:
        save_json(root/"resolved"/(label+".json"),dict(source=str(dest),shard=str(work/"input"/label),
                  source_sha256=digest(dest/"run_manifest.json"),
                  reference_sha256=digest(work/"input"/label/"v22_reference_input_core_summary.json")))
    intake=json.loads((dest/"acquisition_summary.json").read_text())
    return dict(core=label,quadrature_sqdeg=spec.get("time_area"),raw_bytes=sum(p.stat().st_size for p in dest.rglob("*") if p.is_file()),
                rows=intake["normalized_detections"],measurements=ref["measurements"],groups=ref["groups"],
                wall_s=monotonic()-t,rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)


def attempt(task):
    try:return one(task)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return dict(core=task[0],error=type(e).__name__)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ("base","catalogues","sas"):p.add_argument(n,type=Path)
    p.add_argument("--cores",nargs="+");p.add_argument("--workers",type=int,default=4)
    a=p.parse_args()
    if not os.getenv("SLURM_JOB_ID") or not 1<=a.workers<=int(os.environ["SLURM_CPUS_PER_TASK"]):p.error("Use assigned Slurm CPUs")
    root=a.base/"v2.4/registration"
    plan=root/"private_full_footprint_plan.json";sig=digest(plan)
    if sig!=json.loads((root/"v24_footprint_inventory_summary.json").read_text())["signed_manifest_sha256"]:raise ValueError("Changed full footprint registration")
    cells=json.loads(plan.read_text())["new"]
    selected=a.cores or select_pilots(cells)
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        rows=list(pool.map(attempt,[(n,cells[n],a.base,a.catalogues,a.sas,sig) for n in selected]))
    save_json(root/f"v24_intake_{os.environ['SLURM_JOB_ID']}_summary.json",dict(footprint_sha256=sig,rows=rows))
    if any("error" in r for r in rows):raise RuntimeError("Affected cores need inspection")


if __name__=="__main__":main()
