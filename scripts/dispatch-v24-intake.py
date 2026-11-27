"""Dispatch immutable footprint blocks; accounting only, suitable for the login host."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time

ACTIVE={"RUNNING","PENDING","CONFIGURING","COMPLETING","SUSPENDED","RESIZING"}
SCALE=256*3600


def command(*args):
    return subprocess.check_output(args,text=True).strip()


def pending_threads(request):
    """Bound a queued shared CPU allocation before Slurm assigns billed threads."""
    fields=dict(item.split("=",1) for item in request.split(",") if "=" in item)
    units={"":1/1024, "K":1/1024**2, "M":1/1024, "G":1., "T":1024.}
    size=re.fullmatch(r"(\d+(?:\.\d+)?)([KMGT]?)",fields.get("mem","").upper())
    nodes=int(fields.get("node","1"))
    if nodes<1 or not size:
        raise ValueError("Unrecognised pending CPU allocation request; check its memory and node limits")
    gib=float(size[1])*units[size[2]]
    # Use the conservative memory bound used at submission. ReqTRES billing
    # can be one even for a one-thread, 56 GiB job. Actuals take precedence
    # as soon as AllocTRES records the awarded share of the node.
    return min(256*nodes,max(int(fields.get("billing",0)),2*int(fields.get("cpu",0)),math.ceil(2*gib)))


def accounts(work):
    ids=sorted(set((work/"private_allocations.txt").read_text().split()))
    if not ids or not all(k.isdigit() for k in ids):raise ValueError("Invalid Slurm ledger")
    result={}
    for start in range(0,len(ids),100):
        chunk=ids[start:start+100]
        reply=command("sacct","-nPX","-j",",".join(chunk),"-o","JobIDRaw,State,ElapsedRaw,AllocTRES,ReqTRES,TimelimitRaw")
        for line in reply.splitlines():
            key,state,elapsed,alloc,req,limit=line.split("|")[:6]
            def bill(tres):
                m=re.search(r"(?:^|,)billing=(\d+)",tres)
                if m:return int(m.group(1))
                m=re.search(r"(?:^|,)cpu=(\d+)",tres)
                return int(m.group(1)) if m else 0
            live=state.split()[0] in ACTIVE
            if live and not limit.isdigit():raise ValueError("Missing running job wall limit")
            charged=bill(alloc)*int(elapsed)/SCALE
            reserved_threads=(bill(alloc) or pending_threads(req)) if live else 0
            reserve=reserved_threads*int(limit)*60/SCALE if live else charged
            result[key]=(state.split()[0],charged,max(charged,reserve))
    if set(ids)!=set(result):raise RuntimeError("Waiting for complete Slurm accounting")
    return result


def scratch_fraction():
    for line in command("showquota").splitlines():
        if not re.search(r"\|\s*pscratch\s*\|",line):continue
        pct=re.findall(r"([\d.]+)%",line)
        if len(pct)==2:return max(float(v) for v in pct)/100
    raise RuntimeError("Cannot verify PScratch quota")


def summary(path):
    return json.loads(path.read_text())


def save(path,content):
    draft=path.with_suffix(".tmp")
    draft.write_text(json.dumps(content,indent=2)+"\n")
    draft.replace(path)


def block_resources(a, start, end, finished, peak_mib=None):
    # Earlier blocks are bound to the checkout which wrote their acquisition manifests.
    earlier = a.early_launcher is not None and start < a.early_end
    launcher = a.early_launcher if earlier else a.launcher
    memory = a.early_mem if earlier else a.mem
    outstanding = end - start - max((r["completed"] for r in finished), default=0)
    if not 0 < outstanding <= end - start:
        raise ValueError("Invalid partial batch receipt")
    workers = min(a.workers, outstanding)
    if workers < a.workers:
        # Dense remaining cores can be much larger than the median. Retain
        # one measured-memory margin even when only one or two workers remain.
        if peak_mib is not None and peak_mib > 0:
            largest_worker = math.ceil(1.2 * peak_mib / 1024)
            memory = max(largest_worker, math.ceil(memory * workers / a.workers))
    return launcher, workers, memory


def runnable_jobs(active, queue):
    """Only user-held parents have zero runnable workers; their budget stays reserved."""
    held = {job for job, reason in (line.split("|", 1) for line in queue.splitlines())
            if reason == "JobHeldUser"}
    return active - held


def dispatch(a):
    work=a.work;root=work/"registration"
    ledger=work/"private_intake_dispatch.json"
    with (work/"private_dispatch.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (work/"batch-receipts/pause-canonical-intake").exists():
            return "live"
        history=summary(ledger) if ledger.exists() else {}
        prior_transport=set(summary(a.recovered_jobs)) if a.recovered_jobs else set()
        usage=accounts(work)
        if not prior_transport.issubset(usage):
            raise ValueError("Transport recovery contains unaccounted allocations")
        reserved=sum(x[2] for x in usage.values())
        active=set(k for k,v in usage.items() if v[0] in ACTIVE)
        occupied=active
        if getattr(a, "runnable_slots", False):
            queue=command("squeue","-h","-u",os.environ["USER"],"-t","PENDING","-o","%i|%r")
            occupied=runnable_jobs(active,queue)
        slots=a.concurrent-len(occupied & {job for attempts in history.values() for job in attempts})
        capacity=scratch_fraction()
        if capacity>=.95:
            return "quota-review"
        for start in range(a.start,a.end,a.block):
            end=min(start+a.block,a.end)
            label=f"{start}-{end}";past=history.get(label,[])
            if any(job in active for job in past):continue
            if any((root/f"v24_batch_{job}_summary.json").is_file() and
                   summary(root/f"v24_batch_{job}_summary.json")["completed"]==end-start for job in past):continue
            finished=[summary(root/f"v24_batch_{job}_summary.json") for job in past
                      if (root/f"v24_batch_{job}_summary.json").is_file()]
            current=[j for j in past if j not in prior_transport]
            recent=[summary(root/f"v24_batch_{j}_summary.json") for j in current
                    if (root/f"v24_batch_{j}_summary.json").is_file()]
            if len(recent)>=2 and recent[-1]["completed"]==recent[-2]["completed"]:
                continue  # Two current-transport attempts left identical outstanding science.
            if len(current)>=2 and not recent and all(usage[j][0] not in ACTIVE for j in current):
                continue  # Inspect repeated current-transport failures without a batch exit.

            if slots<=0:return "live"
            # Slurm may apportion a larger minimum fraction for memory. Use a
            # conservative reservation for each new request, then use actual TRES.
            progress=root/"v24_intake_progress_summary.json"
            high_water=summary(progress)["peak_individual_rss_mib"] if progress.is_file() else None
            launcher,workers,memory=block_resources(a,start,end,finished,high_water)
            upper=max(workers,2*memory)*a.minutes*60/SCALE
            if reserved+upper>a.ceiling:return "budget-review"
            j=command("sbatch","--parsable","-A",a.account,"-q","shared","-C","cpu","-N1","-n1",
                      "-c",str(workers),"--mem",str(memory)+"G","-t",str(a.minutes),"-J","dpm24-wide",
                      "-o",str(work/"logs/wide-%j.log"),str(launcher),
                      "footprint_batch_v24",str(work.parent),str(a.catalogues),str(a.sas),
                      str(start),str(end),"--workers",str(workers)).split(";")[0]
            if not j.isdigit():raise RuntimeError("Missing Slurm job ID")
            with (work/"private_allocations.txt").open("a") as stream:stream.write(j+"\n")
            history.setdefault(label,[]).append(j);save(ledger,history)
            print("Submitted",label,j,flush=True)
            slots-=1;reserved+=upper
        return "live" if any(j in active for tries in history.values() for j in tries) else "assigned"


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("work","launcher","catalogues","sas"):p.add_argument(key,type=Path)
    p.add_argument("account");p.add_argument("start",type=int);p.add_argument("end",type=int)
    p.add_argument("--workers",type=int,default=10);p.add_argument("--mem",type=int,default=40)
    p.add_argument("--minutes",type=int,default=180);p.add_argument("--block",type=int,default=192)
    p.add_argument("--concurrent",type=int,default=2);p.add_argument("--ceiling",type=float,default=48.)
    p.add_argument("--runnable-slots",action="store_true",help="Count runnable ordinary batches separately from explicitly held dense verification parents")
    p.add_argument("--recovered-jobs",type=Path,help="Frozen accounted job IDs predating a verified archive-transport recovery")
    p.add_argument("--early-launcher",type=Path,help="Immutable checkout used by the first registered blocks")
    p.add_argument("--early-end",type=int,help="First block using the normal launcher")
    p.add_argument("--early-mem",type=int)
    p.add_argument("--watch",action="store_true");a=p.parse_args()
    early=(a.early_launcher,a.early_end,a.early_mem)
    if any(x is not None for x in early) and not all(x is not None for x in early):
        p.error("Specify early launcher, boundary, and memory together")
    if a.early_end is not None and not (a.start<a.early_end<=a.end and (a.early_end-a.start)%a.block==0 and a.early_mem>0):
        p.error("The checkout boundary must coincide with a registered block boundary")
    if not 0<a.ceiling<=48 or a.start<0 or a.start>=a.end:p.error("Invalid cost or footprint bounds")
    while True:
        try:state=dispatch(a)
        except RuntimeError as exc:
            if not a.watch:raise
            print(exc,flush=True);state="live"
        if not a.watch or state!="live":
            print(state,flush=True);break
        time.sleep(30)
