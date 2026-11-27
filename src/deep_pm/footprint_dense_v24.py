"""Stage verified query pages for unusually dense inputs before allocating fitting memory."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import fcntl
import json
import os
from pathlib import Path

from . import access
from .exposure_inputs import joined_pages
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


def stage(task):
    work, label, spec, footprint = task
    receipts = work / "batch-receipts"
    with (receipts / (label + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (receipts / (label + ".json")).is_file():
            return None
        # Canonical replay directories use the same original query cache. The
        # signed science worker will independently verify every SHA and row count.
        cache = access.Cache(work / "intake" / label / "cache")
        halo = SETTINGS["halo_deg"]
        old=work/"dense-pages"/(label+".json")
        if os.getenv("DPM_REQUIRE_DENSE_AUXILIARY")=="1":
            if (receipts/"pause-canonical-intake").exists():
                raise RuntimeError("Confirmed upstream interruption")
            rows=stage_auxiliary(cache,spec,halo)
            save_json(work/"dense-auxiliary"/(label+".json"),dict(core=label,footprint_sha256=footprint,
                      job=os.environ["SLURM_JOB_ID"],**rows))
            if old.is_file():
                proof=json.loads(old.read_text())
                if proof["core"]!=label or proof["footprint_sha256"]!=footprint:
                    raise ValueError("Staged measurement pages belong to a different science input")
                # Normalization rechecks every page against its immutable cache SHA.
                return proof
        south, north = spec["dec"]
        columns = "id AS objectid,ra,dec,pmra,pmdec,pmraerr,pmdecerr,deltamjd ,ndet"
        archive = access.checked_spatial(cache, "nsc", columns, "nsc_dr2.object",
                                        where_box(spec, halo), south-halo-1e-7, north+halo+1e-7)
        if archive.objectid.duplicated().any():
            raise ValueError("Duplicate physical archive IDs")
        rows = pages = 0
        for page in joined_pages(cache,archive,spec,halo,"nsc",1,SETTINGS["native_row_cap"],True,True):
            rows += len(page)
            pages += 1
        result = dict(core=label, footprint_sha256=footprint, delivered_measurements=rows,
                      archive_objects=len(archive), checked_pages=pages,
                      expected_from_archive=int(archive.ndet.sum()), job=os.environ["SLURM_JOB_ID"])
        save_json(work / "dense-pages" / (label + ".json"), result)
        return result


def run(base, start, end, workers):
    work = base / "v2.4"
    plan = work / "registration/private_full_footprint_plan.json"
    sha = digest(plan)
    if sha != json.loads((work / "registration/v24_footprint_inventory_summary.json").read_text())["signed_manifest_sha256"]:
        raise ValueError("Changed footprint before page staging")
    cells = json.loads(plan.read_text())["new"]
    names = [n for group in groups(cells)[1:] for n in group][start:end]
    names = [n for n in names if not (work/"batch-receipts"/(n+".json")).exists()]
    # Start long transfers early, using archived metadata without examining PM.
    pilot = work / "registration/private_density_tail_receipt.json"
    loads = dict(json.loads(pilot.read_text()).get(str(start), [])) if pilot.exists() else {}
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
              dict(footprint=sha, ordinal_range=[start,end], names=names, failed=failures))
    if failures:
        raise RuntimeError("Some dense cores still require successful query pages")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base",type=Path)
    parser.add_argument("start",type=int)
    parser.add_argument("end",type=int)
    parser.add_argument("--workers",type=int,required=True)
    a=parser.parse_args()
    if not os.getenv("SLURM_JOB_ID") or not 1<=a.workers<=int(os.environ["SLURM_CPUS_PER_TASK"]):
        parser.error("Use assigned Slurm CPUs")
    run(a.base,a.start,a.end,a.workers)
