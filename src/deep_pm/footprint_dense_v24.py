"""Stage verified query pages for unusually dense inputs before allocating fitting memory."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import fcntl
import json
import os
from pathlib import Path

from . import access
from .exposure_inputs import confirm_unmeasured, joined_pages
from .exposure_preflight import where_box
from .footprint_intake_v24 import SETTINGS
from .footprint_reference_packs_v24 import groups
from .io import digest, save_json


def stage_auxiliary(cache,spec,halo):
    """Populate the two canonical post-measurement queries before reserving dense memory."""
    from .regional_inputs import GAIA_COLUMNS
    south,north=spec["dec"]
    padding=halo+.65
    chips=access.checked_spatial(cache,"nsc","exposure,ccdnum,instrument,airmass","nsc_dr2.chip",
                                 where_box(spec,padding),south-padding-1e-7,north+padding+1e-7)
    if chips.drop_duplicates().duplicated(["exposure","ccdnum"]).any():
        raise ValueError("Inconsistent physical detector metadata")
    gaia=access.checked_spatial(cache,"nsc",GAIA_COLUMNS+",visibility_periods_used","gaia_dr3.gaia_source",
                                where_box(spec,halo),south-halo-1e-7,north+halo+1e-7)
    return dict(chip_rows=len(chips),gaia_validation_rows=len(gaia))


def stage(task, *, prepare_absences_only=False):
    work, label, spec, footprint = task
    receipts = work / "batch-receipts"
    with (receipts / (label + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (receipts / (label + ".json")).is_file():
            return None
        cache = access.Cache(work / "intake" / label / "cache", read_only=prepare_absences_only)
        halo = SETTINGS["halo_deg"]
        pages_path = work / "dense-pages" / (label + ".json")
        missing_path = work / "dense-absence" / (label + ".json")
        aux_path = work / "dense-auxiliary" / (label + ".json")
        previous = json.loads(pages_path.read_text()) if pages_path.is_file() else None
        if previous and (previous["core"] != label or previous["footprint_sha256"] != footprint):
            raise ValueError("Staged measurement pages belong to another footprint")
        if prepare_absences_only and not previous:
            raise ValueError("A complete staged measurement receipt is required for offline auditing")
        if not prepare_absences_only and (receipts / "pause-canonical-intake").exists():
            raise RuntimeError("Confirmed upstream interruption")
        need_aux = os.getenv("DPM_REQUIRE_DENSE_AUXILIARY") == "1"
        absent = json.loads(missing_path.read_text()) if missing_path.exists() else None
        if absent and (not previous or absent["core"] != label or absent["footprint_sha256"] != footprint
                       or absent["page_receipt_sha256"] != digest(pages_path)):
            raise ValueError("Unmeasured archive population belongs to another page receipt")
        if previous and absent:
            if prepare_absences_only or not need_aux:
                return previous
            if aux_path.is_file():
                aux = json.loads(aux_path.read_text())
                if (aux["core"] == label and aux["footprint_sha256"] == footprint
                        and aux.get("absence_receipt_sha256") == digest(missing_path)):
                    return previous

        if not absent:
            south, north = spec["dec"]
            columns = "id AS objectid,ra,dec,pmra,pmdec,pmraerr,pmdecerr,deltamjd ,ndet"
            archive = access.checked_spatial(cache, "nsc", columns, "nsc_dr2.object",
                                            where_box(spec, halo), south-halo-1e-7, north+halo+1e-7)
            if archive.objectid.isna().any() or archive.objectid.duplicated().any():
                raise ValueError("Invalid physical archive IDs")
            missing = set(archive.objectid.astype(str))
            rows = pages = 0
            # Older complete staging uses ordinary signed scalar counts; no new grouped
            # archive request is necessary to inventory those already accepted bytes.
            grouped = not previous or previous.get("group_count_verified", False)
            for page in joined_pages(cache, archive, spec, halo, "nsc", 1, SETTINGS["native_row_cap"],
                                     True, True, allow_grouped=grouped):
                missing.difference_update(page.objectid.unique().astype(str))
                rows += len(page)
                pages += 1
            facts = dict(core=label, footprint_sha256=footprint, delivered_measurements=rows,
                         archive_objects=len(archive), checked_pages=pages, expected_from_archive=int(archive.ndet.sum()))
            if previous:
                if any(previous.get(k) != v for k,v in facts.items()):
                    raise ValueError("The staged archive population changed on replay")
            else:
                previous = dict(**facts, job=os.environ["SLURM_JOB_ID"],
                                group_count_verified=hasattr(cache, "grouped_verified_rows"))
                save_json(pages_path, previous)
            absent = dict(core=label, footprint_sha256=footprint, page_receipt_sha256=digest(pages_path),
                          missing=sorted(missing))
            save_json(missing_path, absent)
        if need_aux and not prepare_absences_only:
            metadata = stage_auxiliary(cache, spec, halo)
            confirm_unmeasured(cache, "nsc", absent["missing"])
            save_json(aux_path, dict(core=label, footprint_sha256=footprint, job=os.environ["SLURM_JOB_ID"],
                                     absence_receipt_sha256=digest(missing_path),
                                     unmeasured_archive_objects=len(absent["missing"]), **metadata))
        return previous

def priority_names(path, expected_sha, footprint, ordered):
    if digest(path) != expected_sha:
        raise ValueError("Changed priority scheduling plan")
    proof = json.loads(path.read_text())
    if proof["footprint"] != footprint:
        raise ValueError("Different registered footprint")
    names = []
    for row in proof["priority"]:
        ordinal = row["ordinal"]
        if not 0 <= ordinal < len(ordered) or ordered[ordinal] != row["core"]:
            raise ValueError("Priority core disagrees with its original footprint ordinal")
        names.append(row["core"])
    if len(names) != len(set(names)):
        raise ValueError("Duplicate priority core")
    return names


def run(base, start, end, workers, priority_plan=None, priority_sha=None):
    work = base / "v2.4"
    plan = work / "registration/private_full_footprint_plan.json"
    sha = digest(plan)
    if sha != json.loads((work / "registration/v24_footprint_inventory_summary.json").read_text())["signed_manifest_sha256"]:
        raise ValueError("Changed footprint before page staging")
    cells = json.loads(plan.read_text())["new"]
    ordered = [n for group in groups(cells)[1:] for n in group]
    if priority_plan is not None:
        ordered = priority_names(priority_plan, priority_sha, sha, ordered)
    if not 0 <= start < end <= len(ordered):
        raise ValueError("Invalid page-staging range")
    names = ordered[start:end]
    names = [n for n in names if not (work/"batch-receipts"/(n+".json")).exists()]
    # Start long transfers early, using archived metadata without examining PM.
    pilot = work / "registration/private_density_tail_receipt.json"
    loads = dict(json.loads(pilot.read_text()).get(str(start), [])) if pilot.exists() and priority_plan is None else {}
    names.sort(key=lambda n: (-loads.get(n,0), n))
    out = work / "dense-pages"
    out.mkdir(exist_ok=True)
    tasks = [(work,n,cells[n],sha) for n in names]
    failures = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [(n,pool.submit(stage,t)) for n,t in zip(names,tasks)]
        for n,f in futures:
            try:
                f.result()
            except Exception as exc:
                print(n,repr(exc),flush=True)
                failures.append(n)
    save_json(out / ("private_batch_"+os.environ["SLURM_JOB_ID"]+".json"),
              dict(footprint=sha, ordinal_range=[start,end], priority_plan_sha256=priority_sha, names=names, failed=failures))
    if failures:
        raise RuntimeError("Some dense cores still require successful query pages")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base",type=Path)
    parser.add_argument("start",type=int)
    parser.add_argument("end",type=int)
    parser.add_argument("--workers",type=int,required=True)
    parser.add_argument("--priority-plan", type=Path, help="Scheduling subset keyed to the original ordinals")
    parser.add_argument("--priority-sha")
    a=parser.parse_args()
    if bool(a.priority_plan) != bool(a.priority_sha):
        parser.error("Supply both the priority plan and its registered SHA256")
    if not os.getenv("SLURM_JOB_ID") or not 1<=a.workers<=int(os.environ["SLURM_CPUS_PER_TASK"]):
        parser.error("Use assigned Slurm CPUs")
    run(a.base,a.start,a.end,a.workers,a.priority_plan,a.priority_sha)
