"""Account for every planned C0 core, including gaps and unsupported fits."""
import argparse
import json
from pathlib import Path

from .__main__ import tomllib
from .continuous_reference import FILES, verify_directory
from .continuous_summary import area
from .io import digest, save_json
from .public_summary import check_aggregate

FIT_FILES = ("continuous_core_summary.json", "private_qso_blocks.json",
             "private_group_deletions.json", "private_catalogue.csv.gz")


def planned_cells(box, specs, names):
    """Check the exact 1-degree tessellation, including cores formed by merges."""
    west, east = map(int, box["ra"])
    south, north = map(int, box["dec"])
    expected = {(x, y) for x in range(west, east) for y in range(south, north)}
    owners = {}
    occupied = set()
    for name in names:
        bounds = specs[name]
        a, b, c, d = (*bounds["ra"], *bounds["dec"])
        if any(v != int(v) for v in (a, b, c, d)):
            raise ValueError(f"Non-grid core {name}")
        group = {(x, y) for x in range(int(a), int(b)) for y in range(int(c), int(d))}
        if name in owners or not group or not group <= expected or group & occupied:
            raise ValueError(f"Invalid or overlapping planned core {name}")
        owners[name] = group
        occupied.update(group)
    if occupied != expected:
        raise ValueError("Planned cores do not tile the frozen box")
    return owners


def stage_state(directory, name, outputs):
    path = directory / "run_manifest.json"
    if not path.is_file():
        return "missing_receipt"
    attempts = json.loads(path.read_text())["stages"]
    if name is None:
        for stage in attempts.values():
            recorded = stage[-1]
            if recorded["state"] == "completed" and set(outputs) <= recorded["outputs"].keys():
                latest = recorded
                break
        else:
            if not attempts:
                return "missing_stage"
            latest = next(reversed(attempts.values()))[-1]
    elif not attempts.get(name):
        return "missing_stage"
    else:
        latest = attempts[name][-1]
    if latest["state"] != "completed":
        return "failed" if latest["state"] == "failed" else "incomplete_attempt"
    recorded = latest["outputs"]
    return ("recorded" if all(recorded.get(k) and (directory / k).is_file() for k in outputs)
            else "missing_output")


def audit(box, specs, plan, outputs, shared):
    cells = planned_cells(box, specs, (name for name, _, _ in plan))
    offsets = shared / "private_c0.npz"
    graph = digest(offsets) if offsets.is_file() else None
    fit_paths = {}
    status = {}
    area_by_state = {}
    for name, source, ref in plan:
        work = outputs / name
        record = {
            "acquisition": stage_state(source, None, FILES),
            "reference": stage_state(ref, "continuous-ref-input-v12", ("reference.csv.gz", "reference_summary.json")),
            "fit": stage_state(work, "continuous-core-v12", FIT_FILES),
        }
        if all(v == "recorded" for v in record.values()):
            try:
                verify_directory(work, FIT_FILES)
                fit = json.loads((work / FIT_FILES[0]).read_text())
                if graph is None or fit["shared_solution_sha256"] != graph:
                    record["fit"] = "wrong_graph"
                else:
                    fit_paths[name] = str(work)
                    record["fit"] = "verified" if fit["reference_gauge"] is not None else "verified_no_gauge"
                    record["gauge"] = fit["reference_gauge"]
            except (ValueError, OSError):
                record["fit"] = "unverified_output"
        record["cells"] = len(cells[name])
        record["area_sqdeg"] = area(specs[name])
        status[name] = record
        state = record["fit"]
        area_by_state[state] = area_by_state.get(state, 0.0) + record["area_sqdeg"]
    result = {
        "target_area_sqdeg": area(box),
        "planned_cells": sum(map(len, cells.values())),
        "planned_core_count": len(cells),
        "verified_core_count": len(fit_paths),
        "verified_cells": sum(len(cells[k]) for k in fit_paths),
        "area_by_fit_state_sqdeg": area_by_state,
        "unverified_cores": [k for k in cells if k not in fit_paths],
        "cores": status,
    }
    check_aggregate(result)
    return result, fit_paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("state", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("--prepare-summary", action="store_true")
    args = parser.parse_args()
    plan = [(name, Path(source), Path(ref)) for name, source, ref in
            (line.split("\t") for line in (args.state / "run-plan.tsv").read_text().splitlines())]
    conf = tomllib.loads(args.config.read_text())
    shared = args.state / "shared"
    result, fits = audit(conf["boxes"]["c"], conf["fields"], plan, args.state / "outputs", shared)
    save_json(args.state / "v12_coverage_audit_summary.json", result)
    if args.prepare_summary:
        save_json(args.state / "fits.json", fits)
        save_json(args.state / "gauges.json", {"all": str(shared)})
        if not fits:
            parser.exit(2, "No verified W0 fits; coverage audit was saved.\n")


if __name__ == "__main__":
    main()
