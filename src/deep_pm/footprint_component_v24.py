"""Export and solve one or more complete, independently connected physical components."""
import argparse
import json
import os
from pathlib import Path
from time import monotonic

from .full_reference_graph import export_component
from .full_reference_run import solve
from .io import digest, save_json


def read(path):
    return json.loads(path.read_text())


def same_archived_input(current, old):
    # The component index fixes the original RNG seed. Source index ordering,
    # precision tensors, centred positions, names and counts must all match.
    required = ("component", "center_deg", "measurements", "groups", "reference_sources")
    return (all(current["component"][k] == old["component"][k] for k in required)
            and current["files"] == old["files"])


def reference(root, number, memory, threads, workers):
    start = monotonic()
    dest = root/"reference"/f"component-{number:04d}"
    export_component(root, number, memory=memory, threads=threads)
    numeric = dest/"private_numeric_binding.json"
    signature = digest(numeric)
    receipt = dest/"v22_reference_component_summary.json"
    if receipt.exists():
        result = read(receipt)
        if result["numeric_binding_sha256"] != signature or digest(dest/"private_c0_unfiltered.npz") != result["graph_sha256"]:
            raise ValueError("Changed completed physical reference solution")
        return
    current = read(numeric)
    registered = read(root/"registration/private_full_coordinate_plan.json")
    for path, spec in registered["graphs"].items():
        graph = Path(path)
        previous = graph.parent/"private_numeric_binding.json"
        if not previous.exists() or not same_archived_input(current, read(previous)):
            continue
        report = read(graph.parent/"v22_reference_component_summary.json")
        solution = read(graph.parent/"private_solution_binding.json")
        if (digest(previous) != report["numeric_binding_sha256"]
                or solution["numeric_sha256"] != digest(previous)
                or solution["bootstrap_draws"] != 48 or solution["seed"] != 45116+number
                or digest(graph) != spec["sha256"] or report["graph_sha256"] != spec["sha256"]):
            raise ValueError("Changed archived physical reference provenance")
        for name in ("private_c0.npz","private_c0_unfiltered.npz"):
            dest_file=dest/name
            origin=(graph.parent/name).resolve(strict=True)
            if dest_file.exists() and dest_file.resolve()!=origin:
                raise ValueError("An independent solution is already present")
            if not dest_file.exists():
                dest_file.symlink_to(origin)
        save_json(dest/"private_archived_solution_binding.json",dict(graph_sha256=spec["sha256"],
                  old_numeric_sha256=digest(previous), current_numeric_sha256=signature,
                  old_solution_sha256=digest(graph.parent/"private_solution_binding.json")))
        result=dict(report, numeric_binding_sha256=signature, reused_archived_reference=True,
                    previous_elapsed_seconds=report["elapsed_seconds"], elapsed_seconds=monotonic()-start,
                    slurm_job=os.environ["SLURM_JOB_ID"])
        save_json(receipt,result)
        return
    solve(root, number, draw_workers=workers)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root",type=Path)
    parser.add_argument("components",nargs="+",type=int)
    parser.add_argument("--threads",type=int,required=True)
    parser.add_argument("--memory",required=True)
    parser.add_argument("--draw-workers",type=int,required=True)
    a=parser.parse_args()
    cpus=int(os.getenv("SLURM_CPUS_PER_TASK",0))
    if not os.getenv("SLURM_JOB_ID") or not 1<=a.draw_workers<=cpus or not 1<=a.threads<=cpus:
        parser.error("Use allocated Slurm CPUs")
    if a.draw_workers>1:
        checked=read(a.root.parent/"registration/v24_parallel_draw_check_summary.json")
        if not checked["byte_equal_values"] or checked["workers"]<a.draw_workers:
            raise ValueError("No admitted real parallel draw equivalence at this worker count")
    for component in a.components:
        reference(a.root,component,a.memory,a.threads,a.draw_workers)
