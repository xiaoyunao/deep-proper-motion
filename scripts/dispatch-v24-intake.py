"""Dispatch immutable footprint blocks; accounting only, suitable for the login host."""
import argparse
import fcntl
import json
from pathlib import Path
import re
import subprocess
import time

ACTIVE={"RUNNING","PENDING","CONFIGURING","COMPLETING","SUSPENDED","RESIZING"}
SCALE=256*3600


def command(*args):
    return subprocess.check_output(args,text=True).strip()


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
            reserve=(max(bill(req),bill(alloc))*int(limit)*60/SCALE if live else charged)
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


def dispatch(a):
    work=a.work;root=work/"registration"
    ledger=work/"private_intake_dispatch.json"
    with (work/"private_dispatch.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        history=summary(ledger) if ledger.exists() else {}
        usage=accounts(work)
        reserved=sum(x[2] for x in usage.values())
        active=set(k for k,v in usage.items() if v[0] in ACTIVE)
        slots=a.concurrent-sum(job in active for attempts in history.values() for job in attempts)
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
            if len(finished)>=2 and finished[-1]["completed"]==finished[-2]["completed"]:
                continue  # No scientific progress on two completed attempts. Inspect affected cores.
            if len(past)>=2 and not finished and all(usage[j][0] not in ACTIVE for j in past):
                continue  # Two failed parents without a verified batch exit need resource diagnosis.

            if slots<=0:return "live"
            # Slurm may apportion a larger minimum fraction for memory. Use a
            # conservative reservation for each new request, then use actual TRES.
            upper=max(a.workers,2*a.mem)*a.minutes*60/SCALE
            if reserved+upper>a.ceiling:return "budget-review"
            j=command("sbatch","--parsable","-A",a.account,"-q","shared","-C","cpu","-N1","-n1",
                      "-c",str(a.workers),"--mem",str(a.mem)+"G","-t",str(a.minutes),"-J","dpm24-wide",
                      "-o",str(work/"logs/wide-%j.log"),str(a.launcher),
                      "footprint_batch_v24",str(work.parent),str(a.catalogues),str(a.sas),
                      str(start),str(end),"--workers",str(a.workers)).split(";")[0]
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
    p.add_argument("--watch",action="store_true");a=p.parse_args()
    if not 0<a.ceiling<=48 or a.start<0 or a.start>=a.end:p.error("Invalid cost or footprint bounds")
    while True:
        try:state=dispatch(a)
        except RuntimeError as exc:
            if not a.watch:raise
            print(exc,flush=True);state="live"
        if not a.watch or state!="live":
            print(state,flush=True);break
        time.sleep(30)
