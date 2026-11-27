"""Dynamically pack registered windows onto allocated CPUs, with bounded retries."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import fcntl
import json
import os
from pathlib import Path
from time import monotonic

from .footprint_intake_v24 import attempt,select_pilots
from .io import digest,save_json


def worker(job):
    tasks,control=job
    done=skipped=failed=0
    for task in tasks:
        core=task[0];receipt=control/(core+".json")
        if receipt.exists():continue
        if stopped(control):break
        with (control/(core+".lock")).open("a") as guard:
            try:fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:continue
            if receipt.exists():continue
            error=control/(core+".failed.json")
            failures=json.loads(error.read_text()) if error.is_file() else []
            rule="canonical" if os.environ.get("DPM_NSC_TRANSPORT","native")=="native" else "canonical-counted-tap"
            if sum(f.get("rule")==rule for f in failures)>=2 or any(f["allocation"]==os.environ["SLURM_JOB_ID"] for f in failures):
                skipped+=1
                continue
            result=attempt(task)
            if "error" in result:
                failed+=1
                failures.append(dict(rule=rule,allocation=os.environ["SLURM_JOB_ID"],error=result["error"]))
                save_json(error,failures)
            else:
                result["manifest_sha256"]=task[-1]
                root=task[2]/"v2.4"
                resolved=root/"resolved"/(core+".json")
                if resolved.is_file():
                    resolved_paths=json.loads(resolved.read_text())
                    result["input_receipt_sha256"]=resolved_paths["source_sha256"]
                    result["reference_receipt_sha256"]=resolved_paths["reference_sha256"]
                else:
                    result["input_receipt_sha256"]=digest(root/"intake"/core/"run_manifest.json")
                    result["reference_receipt_sha256"]=digest(root/"input"/core/"v22_reference_input_core_summary.json")
                save_json(receipt,result)
                done+=1
    return dict(completed=done,failed=failed,previously_failed=skipped)


def stopped(control):
    # The supervisor may stop assignment before a soft user quota is reached.
    return (control/"pause-canonical-intake").exists()


def run(base,catalogues,sas,start,end,workers,archive_boundaries=False):
    root=base/"v2.4";registration=root/"registration"
    path=registration/"private_full_footprint_plan.json";sig=digest(path)
    if sig!=json.loads((registration/"v24_footprint_inventory_summary.json").read_text())["signed_manifest_sha256"]:
        raise ValueError("Footprint signature mismatch")
    cells=json.loads(path.read_text())["new"]
    pilots=set(select_pilots(cells))
    def order(name):
        box=cells[name];r=sum(box["ra"])/2;d=sum(box["dec"])/2
        stripe=abs(d)<1.6 and (r<53 or r>=307)
        return (not stripe,r,d,name)
    names=sorted(cells.keys()-pilots,key=order)[start:end]
    if archive_boundaries:
        audit=root/"meridian/private_archive_audit.json"
        proof=json.loads((root/"meridian/v24_archive_input_audit_summary.json").read_text())
        if proof["private_audit_sha256"]!=digest(audit):raise ValueError("Changed archive geometry audit")
        boundary=json.loads(audit.read_text())
        names=[n for n in names if n in boundary and not boundary[n]["old"]]
    control=root/"batch-receipts";control.mkdir(exist_ok=True)
    tasks=[(c,cells[c],base,catalogues,sas,sig) for c in names]
    # Blocks are visited in a different cyclic order. Lock stealing keeps
    # long queries in a dense field from idling all other assigned workers.
    count=min(workers,len(tasks))
    def cyclic(k):
        ix=len(tasks)*k//count
        return tasks[ix:]+tasks[:ix]
    t=monotonic()
    results=[]
    if count:
        with ProcessPoolExecutor(max_workers=count) as pool:
            results=list(pool.map(worker,[(cyclic(k),control) for k in range(count)]))
    save_json(registration/f"v24_batch_{os.environ['SLURM_JOB_ID']}_summary.json",
              dict(manifest_sha256=sig,ordinal_range=[start,end],selected=len(names),
                   completed=sum((control/(c+".json")).is_file() for c in names),archive_boundary_subset=archive_boundaries,workers=results,
                   elapsed_s=monotonic()-t))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for n in ("base","catalogues","sas"):p.add_argument(n,type=Path)
    p.add_argument("start",type=int);p.add_argument("end",type=int);p.add_argument("--workers",type=int,required=True)
    p.add_argument("--archive-boundaries",action="store_true",help="Prioritise registered meridian audit cores under the ordinary per-core locks")
    a=p.parse_args()
    if not os.getenv("SLURM_JOB_ID") or not 1<=a.workers<=int(os.environ["SLURM_CPUS_PER_TASK"]):p.error("Use assigned Slurm CPUs")
    if not 0<=a.start<a.end:p.error("Invalid registered ordinal range")
    run(**vars(a))
