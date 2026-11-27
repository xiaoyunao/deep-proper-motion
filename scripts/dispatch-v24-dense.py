"""Finish a dense tail against its real page counts and release its original held batch."""
import argparse
import fcntl
import math
from pathlib import Path
import runpy
import time

ops = runpy.run_path(str(Path(__file__).with_name("dispatch-v24-intake.py")))
read = ops["summary"]


def advance(a):
    w=a.work
    history_path=w/f"private_dense_dispatch_{a.start}_{a.end}.json"
    with (w/"private_dispatch.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        history=read(history_path) if history_path.is_file() else {}
        if history.get("released"):
            return "released"
        regular=read(w/"private_intake_dispatch.json")[f"{a.start}-{a.end}"]
        if a.held_job not in regular:
            raise ValueError("Held parent belongs to a different registered range")
        states=ops["accounts"](w)
        warm_state=states[a.warm_job][0]
        if warm_state not in ops["ACTIVE"] and warm_state!="COMPLETED":
            raise ValueError("Page staging ended without complete receipts; inspect its failed cores")
        # This list contains only observed exceptional cores from the registered
        # range. The signed per-core page count determines the actual allocation.
        observed=read(w/"registration/private_density_tail_receipt.json")[str(a.start)]
        large={n for n,count in observed if count>=4_000_000}
        if warm_state=="COMPLETED":
            warm=read(w/"dense-pages"/("private_batch_"+a.warm_job+".json"))
            if warm["ordinal_range"]!=[a.start,a.end] or warm["failed"]:
                raise ValueError("The staged ordinals did not close")
            for n in warm["names"]:
                proof=w/"dense-pages"/(n+".json")
                if proof.is_file() and read(proof)["delivered_measurements"]>=4_000_000:
                    large.add(n)
        active=sum(states[j][0] in ops["ACTIVE"] for j in history.values())
        bound=sum(v[2] for v in states.values())
        pending=0
        for n in sorted(large):
            if (w/"batch-receipts"/(n+".json")).is_file():
                continue
            pending+=1
            if n in history:
                j=history[n]
                if states[j][0] not in ops["ACTIVE"]:
                    raise ValueError(f"{n}: {j} ended {states[j][0]} without a science receipt")
                continue
            proof=w/"dense-pages"/(n+".json")
            if not proof.exists() or active>=a.concurrent:
                continue
            staged=read(proof)
            if staged["footprint_sha256"]!=read(w/"registration/v24_footprint_inventory_summary.json")["signed_manifest_sha256"] or staged["core"]!=n:
                raise ValueError("Changed dense paging footprint")
            # The observed 12.62M-row core reached 25 GiB in a worker. Reserve
            # 2700 bytes per row plus loader overhead. The request covers only
            # cached processing, not the preceding network transfer.
            rows=staged["delivered_measurements"]
            gib=max(16,math.ceil(3+2700*rows/2**30))
            minutes=120
            if gib>450:
                raise ValueError("One dense core exceeds the CPU node memory envelope")
            cost=2*gib*minutes/(256*60)
            if bound+cost>48 or ops["scratch_fraction"]()>=.95:
                return "resource-review"
            job=ops["command"]("sbatch","--parsable","-A",a.account,"-q","shared","-C","cpu",
                              "-N1","-n1","-c1","--mem",f"{gib}G","-t",str(minutes),"-J","dpm24-dense",
                              "-o",str(w/"logs/dense-%j.log"),str(a.launcher),n).split(";")[0]
            if not job.isdigit():
                raise RuntimeError("Missing dense Slurm job ID")
            with (w/"private_allocations.txt").open("a") as f:
                f.write(job+"\n")
            history[n]=job
            ops["save"](history_path,history)
            print("Submitted",n,rows,gib,job,flush=True)
            active+=1;bound+=cost
        if pending or warm_state!="COMPLETED":
            return "running"
        held_state=states[a.held_job][0]
        if held_state=="PENDING":
            reason=ops["command"]("squeue","-h","-j",a.held_job,"-o","%r")
            if reason=="JobHeldUser":
                ops["command"]("scontrol","release",a.held_job)
            elif reason.startswith("JobHeld"):
                raise ValueError(f"Unexpected Slurm hold: {reason}")
        elif held_state not in ("RUNNING","COMPLETING","COMPLETED"):
            raise ValueError(f"Original dense batch ended {held_state}")
        history["released"]=a.held_job
        ops["save"](history_path,history)
        return "released"


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for n in ("work","launcher"):
        p.add_argument(n,type=Path)
    p.add_argument("account");p.add_argument("warm_job");p.add_argument("held_job")
    p.add_argument("start",type=int);p.add_argument("end",type=int)
    p.add_argument("--concurrent",type=int,default=3)
    a=p.parse_args()
    if a.concurrent<1:
        p.error("Set positive concurrency")
    while True:
        try:
            state=advance(a)
        except RuntimeError as exc:
            if str(exc)!="Waiting for complete Slurm accounting":
                raise
            state="running"
        if state!="running":
            print(state,flush=True)
            break
        time.sleep(45)
