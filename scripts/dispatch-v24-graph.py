"""Advance signed whole-footprint acquisition to the frozen physical reference graph."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import runpy
from time import sleep

intake = runpy.run_path(str(Path(__file__).with_name("dispatch-v24-intake.py")))


def graph_step(work, catalogues, sas, history):
    total = intake["summary"](work/"registration/v24_footprint_inventory_summary.json")["new_rectangles"]
    pilot_names = set()
    for path in (work/"registration").glob("v24_intake_*_summary.json"):
        if path.name.split("_")[2].isdigit():
            pilot_names.update(r["core"] for r in intake["summary"](path)["rows"] if "error" not in r)
    signed = {name[:-5] for name in os.listdir(work/"batch-receipts")
              if name.endswith(".json") and not name.endswith(".failed.json")}
    if len(signed | pilot_names) < total:
        return None
    base = str(work.parent)
    audit_path = work/"meridian/v24_archive_input_audit_summary.json"
    record = intake["summary"](audit_path)
    if "final-archive-audit" not in history:
        return "final-archive-audit", 1, 8, 55, ["footprint_longitude_audit_v24", base]
    if record["signed_inputs_checked"] != record["possible"]:
        raise ValueError("Unresolved registered archive inputs after full audit")
    if record["inputs_for_replay"]:
        if "archive-repair" not in history:
            return "archive-repair", 4, 36, 180, ["footprint_repair_v24",base,str(catalogues),str(sas),"--workers","4"]
        if "audit-repair" not in history:
            return "audit-repair", 1, 8, 55, ["footprint_longitude_audit_v24",base]
        raise ValueError("Archive differences remain after the signed repair")
    if not (work/"registration/v24_graph_registration_summary.json").exists():
        return "bind-inputs", 1, 4, 100, ["footprint_reference_v24", base]
    graph = work/"joint/graph/v22_global_graph_summary.json"
    if not graph.exists():
        return "physical-graph", 16, 80, 240, ["full_reference_graph","close",str(work/"joint"),
                                                "--memory","54GB","--threads","16"]
    return "closed"


def dispatch(args):
    work = args.work
    plan = work/"private_reference_dispatch.json"
    with (work/"private_dispatch.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        history = intake["summary"](plan) if plan.exists() else {}
        if history:
            usage = intake["accounts"](work)
            for stage, job in history.items():
                state = usage[job][0]
                if state in intake["ACTIVE"]:
                    return "running"
                if state != "COMPLETED":
                    raise ValueError(f"{stage} {job} ended {state}; inspect the stage and its checkpoints")
        stage = graph_step(work, args.catalogues, args.sas, history)
        if stage is None:
            return "acquiring"
        if stage == "closed":
            return stage
        label, cpus, memory, minutes, command = stage
        if label in history:
            raise ValueError(f"{label} exited without its required scientific receipt")
        total = sum(v[2] for v in intake["accounts"](work).values())
        # This conservative upper bound also covers memory-weighted shared partitions.
        if total + max(cpus,2*memory)*minutes/(256*60) > 48:
            return "budget-review"
        if intake["scratch_fraction"]() >= .95:
            return "quota-review"
        job = intake["command"]("sbatch","--parsable","-A",args.account,"-q","shared","-C","cpu",
                "-N1","-n1","-c",str(cpus),"--mem",str(memory)+"G","-t",str(minutes),
                "-J","dpm24-ref","-o",str(work/"logs/reference-%j.log"),str(args.launcher),*command).split(";")[0]
        if not job.isdigit():
            raise RuntimeError("Missing Slurm job ID")
        with (work/"private_allocations.txt").open("a") as f:
            f.write(job+"\n")
        history[label] = job
        intake["save"](plan, history)
        print("Submitted",label,job,flush=True)
        return "running"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("work","launcher","catalogues","sas"):
        parser.add_argument(name,type=Path)
    parser.add_argument("account")
    parser.add_argument("--watch",action="store_true")
    args = parser.parse_args()
    prior = None
    while True:
        try:
            state = dispatch(args)
        except RuntimeError as exc:
            if not args.watch or str(exc) != "Waiting for complete Slurm accounting":
                raise
            state = "waiting for Slurm accounting"
        if state != prior:
            print(state,flush=True)
            prior = state
        if not args.watch or state in ("closed","budget-review","quota-review"):
            break
        sleep(60)
