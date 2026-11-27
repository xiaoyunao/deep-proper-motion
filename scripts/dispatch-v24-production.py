"""Schedule independent reference components and registered source ranges to catalogue closure."""
import argparse
from dataclasses import dataclass
import fcntl
import math
from pathlib import Path
import runpy
from time import sleep

intake=runpy.run_path(str(Path(__file__).with_name("dispatch-v24-intake.py")))
read=intake["summary"]
assessment=runpy.run_path(str(Path(__file__).with_name("v24_assessment.py")))["assessment"]


@dataclass
class Stage:
    label: str
    args: list
    cpus: int
    gib: int
    minutes: int
    completed: int
    expected: int


def component_batches(components):
    small=[]
    rows=0
    for c in sorted(components,key=lambda c:-c["measurements"]):
        if c["measurements"]>=2_000_000:
            yield [c]
        else:
            if small and (rows+c["measurements"]>5_000_000 or len(small)>=20):
                yield small
                small=[];rows=0
            small.append(c);rows+=c["measurements"]
    if small:
        yield small


def schedule(work,history=None):
    history=history or {}
    root=work/"joint"
    graph=root/"graph/v22_global_graph_summary.json"
    if not graph.exists():
        return "waiting for the complete physical graph",[]
    comps=[c for c in read(graph)["components"] if c["status"]=="ready"]
    if not comps:
        raise ValueError("No cross-survey physical component; inspect the graph result")
    pending=[]
    for group in component_batches(comps):
        numbers=[c["component"] for c in group]
        folders=[root/"reference"/f"component-{n:04d}" for n in numbers]
        done=sum((p/"v22_reference_component_summary.json").exists() for p in folders)
        if done==len(group):
            continue
        size=max(c["measurements"] for c in group)
        memory=max(16,math.ceil(6+size*160/2**30))
        if memory>450:
            raise ValueError("Reference allocation exceeds a CPU node; inspect actual component inputs")
        minutes=min(360,max(45,math.ceil(25+sum(c["measurements"] for c in group)/1e6*1.3)))
        args=["footprint_component_v24",str(root),*map(str,numbers),"--threads","4",
              "--memory",f"{max(8,min(40,memory//2))}GB","--draw-workers","4"]
        # Durable IRLS and per-draw files provide an actual retry progress measure.
        progress=done*64+sum(len(list((p/"checkpoints").glob("*.sha256"))) for p in folders)
        pending.append(Stage("reference-"+"-".join(map(str,numbers)),args,4,memory,minutes,progress,len(group)*128))
    if pending:
        return "reference",pending
    if not (root/"admission/v24_reference_admission_summary.json").exists():
        return "reference admission",[Stage("admission",["footprint_reference_admission_v24",str(root)],2,8,30,0,1)]
    pending=[]
    for c in comps:
        number=c["component"]
        output=root/"held-validation"/f"component-{number:04d}"
        if not (output/"v22_complete_reference_validation_summary.json").is_file():
            pending.append(Stage(f"held-{number}",["full_reference_validation",str(root),str(number),str(output)],
                                 4,24,100,0,1))
    if pending:
        return "held galaxies",pending
    closed=root/"catalogue/v22_catalogue_ownership_summary.json"
    if closed.exists():
        return "catalogue closed",[]
    plan=read(root/"registration/private_full_coordinate_plan.json")["cores"]
    # A block is a scheduling range. Every fit uses all its physical graph groups.
    for start in range(0,len(plan),192):
        end=min(start+192,len(plan))
        label=f"sources-{start}-{end}"
        complete=False
        for attempt in history.get(label,[]):
            proof=root/"batch"/f"component-range-{start}-{end}"/attempt["job"]/"v22_source_batch_summary.json"
            if proof.is_file() and read(proof)["completed_core_checks"]==end-start:
                complete=True
                break
        if complete:
            continue
        done=sum((root/"sources"/e["core"]/"v22_source_core_summary.json").exists() for e in plan[start:end])
        if done==end-start:
            continue
        pending.append(Stage(f"sources-{start}-{end}",["full_coordinate_batch",str(root),"--core-range",str(start),
                 str(end),"--workers","20","--memory-gib","86","--hours","2.55","--memory-model",
                 "receipt-validated"],20,96,180,done,end-start))
    if pending:
        return "sources",pending
    indexed=sum((root/"catalogue/index"/e["core"]/"private_index_receipt.json").exists() for e in plan)
    if indexed<len(plan):
        return "index",[Stage("index",["full_coordinate_catalogue","index",str(root),"--workers","12"],
                             12,68,240,indexed,len(plan))]
    return "ownership",[Stage("ownership",["full_coordinate_catalogue","close",str(root)],8,54,240,0,1)]


def dispatch(args):
    w=args.work
    # Only count core-level files on first transition, or after a job ended.
    # Avoid repeated large Lustre directory walks while a full fleet is working.
    validation = args.spectra is not None
    history_path=w/("private_assessment_dispatch.json" if validation else "private_production_dispatch.json")
    with (w/"private_dispatch.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if validation and not (w/"joint/catalogue/v22_catalogue_ownership_summary.json").exists():
            return "waiting for catalogue ownership"
        if not (w/"joint/graph/v22_global_graph_summary.json").exists():
            return "waiting for the complete physical graph"
        history=read(history_path) if history_path.exists() else {}
        usage=intake["accounts"](w)
        live={j for j,v in usage.items() if v[0] in intake["ACTIVE"]}
        active=sum(a["job"] in live for attempts in history.values() for a in attempts)
        if active>=args.concurrent:
            return "running"
        graph_history=w/"private_reference_dispatch.json"
        if graph_history.is_file():
            graph_job=read(graph_history).get("physical-graph")
            if graph_job and graph_job in live:
                return "waiting for physical graph checkpoint"
            if graph_job and usage[graph_job][0]!="COMPLETED":
                raise ValueError("The physical graph parent did not close successfully")
        stage,items=assessment(w,args.spectra,args.xiao,Stage) if validation else schedule(w,history)
        if not items:
            if active:
                return "running"
            if validation:
                failed=[attempts[-1]["job"] for attempts in history.values()
                        if usage[attempts[-1]["job"]][0]!="COMPLETED"]
                if failed:raise ValueError(f"Inspect outstanding assessment allocations: {failed}")
            return stage
        spent=sum(v[2] for v in usage.values())
        problems=[]
        for item in items:
            past=history.get(item.label,[])
            if any(a["job"] in live for a in past):
                continue
            if past:
                last=past[-1]
                state=usage[last["job"]][0]
                if state not in ("TIMEOUT","COMPLETED") or item.completed<=last["progress"]:
                    problems.append((item.label,last["job"],state))
                    continue
            bound=max(item.cpus,2*item.gib)*item.minutes/(256*60)
            if spent+bound>48:
                return "running" if active else "budget-review"
            if intake["scratch_fraction"]()>=.95:
                return "quota-review"
            job=intake["command"]("sbatch","--parsable","-A",args.account,"-q","shared","-C","cpu",
                    "-N1","-n1","-c",str(item.cpus),"--mem",str(item.gib)+"G","-t",str(item.minutes),
                    "-J","dpm24-"+stage.split()[0],"-o",str(w/"logs/production-%j.log"),
                    str(args.launcher),*item.args).split(";")[0]
            if not job.isdigit():
                raise RuntimeError("Missing Slurm job ID")
            with (w/"private_allocations.txt").open("a") as stream:
                stream.write(job+"\n")
            history.setdefault(item.label,[]).append(dict(job=job,progress=item.completed,expected=item.expected))
            intake["save"](history_path,history)
            print("Submitted",item.label,job,flush=True)
            spent+=bound;active+=1
            if active>=args.concurrent:
                break
        if problems and not active:
            raise ValueError(f"Outstanding numerical or scheduling receipts need inspection: {problems}")
        return "running"


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("work",type=Path);p.add_argument("launcher",type=Path);p.add_argument("account")
    p.add_argument("--spectra",type=Path,help="Post-ownership assessment only, using these frozen spectra")
    p.add_argument("--xiao",type=Path,help="Published Xiao catalogue for the frozen DESI comparison")
    p.add_argument("--concurrent",type=int,default=4);p.add_argument("--watch",action="store_true")
    a=p.parse_args()
    if (a.spectra is None)!=(a.xiao is None):p.error("Specify both external validation inputs")
    if a.concurrent<1:p.error("Set a positive allocation count")
    previous=None
    while True:
        try:state=dispatch(a)
        except RuntimeError as exc:
            if not a.watch or str(exc)!="Waiting for complete Slurm accounting":raise
            state="waiting for Slurm accounting"
        if state!=previous:
            print(state,flush=True);previous=state
        if not a.watch or state in ("catalogue closed","assessment finished","quota-review","budget-review"):
            break
        sleep(60)
