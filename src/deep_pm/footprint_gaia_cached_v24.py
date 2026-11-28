"""Complete cached measurements using a Gaia DR3 positional veto verified on independent detections."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess

from .footprint_intake_v24 import one,select_pilots
from .io import digest,save_json


def evidence(work, core, footprint):
    reg=work/"registration"
    proof=reg/"v24_independent_gaia_veto_summary.json"
    q=json.loads(proof.read_text())
    controls=reg/"v24_gaia_veto_controls_summary.json"
    c=json.loads(controls.read_text())
    stage=work/"gaia-veto-trial"/core
    table=stage/"gaia.csv"
    p=json.loads((stage/"private_provenance.json").read_text())
    if (footprint!=q["footprint"] or footprint!=p["footprint"] or p["core"]!=core
            or not q["reference_file_byte_equal"] or q["original_sha256"]!=q["replayed_sha256"]
            or q["controls_sha256"]!=digest(controls) or p["controls_sha256"]!=q["controls_sha256"]
            or c["footprint"]!=footprint or not c["position_membership_passes"] or len(c["controls"])!=5
            or digest(table)!=p["gaia_sha256"]):
        raise ValueError("Different Gaia positions or missing independent frozen reference verification")
    sha=digest(proof)
    config=(f'\ngaia_veto = "original-dr3-csv-position-only"\ngaia_veto_proof_sha256 = "{sha}"'
            f'\ngaia_veto_input_sha256 = "{p["gaia_sha256"]}"')
    return dict(file=table,proof_sha=sha,config=config)


def run(base,catalogues,sas,cores):
    work=base/"v2.4"
    manifest=work/"registration/private_full_footprint_plan.json"
    fp=digest(manifest)
    reg=json.loads((work/"registration/v24_footprint_inventory_summary.json").read_text())
    if fp!=reg["signed_manifest_sha256"] or not os.getenv("DPM_SDSS_STAGE"):
        raise ValueError("Require the frozen footprint and staged SDSS input")
    cells=json.loads(manifest.read_text())["new"]
    pilots=set(select_pilots(cells))
    if set(cores)&pilots or len(set(cores))!=len(cores):
        raise ValueError("A closed original pilot or repeated core was supplied")
    commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    results=[]
    controls=work/"batch-receipts"
    for name in cores:
        target=controls/(name+".json")
        with (controls/(name+".lock")).open("a") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if target.exists():
                if json.loads(target.read_text())["manifest_sha256"]!=fp:
                    raise ValueError("Existing source footprint changed")
                results.append(dict(status="previously_signed"))
                continue
            g=evidence(work,name,fp)
            stage=work/"sdss-stage"/name
            if not stage.is_dir():
                raise ValueError("Early positions are not staged for this footprint")
            revision="gaia-position-"+commit[:12]+"-"+g["proof_sha"][:12]
            try:
                r=one((name,cells[name],base,catalogues,sas,fp),revision,gaia_veto=g)
            except FileNotFoundError as exc:
                if "No intact local response for " not in str(exc):
                    raise
                results.append(dict(status="missing_archive_response"))
                continue
            source=work/"revisions"/revision/"intake"/name
            ref=work/"revisions"/revision/"input"/name/"v22_reference_input_core_summary.json"
            r.update(manifest_sha256=fp,input_receipt_sha256=digest(source/"run_manifest.json"),
                     reference_receipt_sha256=digest(ref))
            save_json(target,r)
            results.append(dict(status="signed",rows=r["rows"],reference_measurements=r["measurements"],
                                groups=r["groups"],time_area_sqdeg=r["quadrature_sqdeg"]))
    save_json(work/"registration"/("v24_gaia_cached_"+os.environ["SLURM_JOB_ID"]+"_summary.json"),
              dict(footprint=fp,job=os.environ["SLURM_JOB_ID"],release=commit,results=results,
                   external_measurement_requests=0,scope="Published original Gaia DR3 positions used only for the reference veto."))
    print(results,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("base","catalogues","sas"):
        p.add_argument(key,type=Path)
    p.add_argument("cores",nargs="+")
    a=p.parse_args()
    if not os.getenv("SLURM_JOB_ID"):p.error("Use Slurm")
    run(a.base,a.catalogues,a.sas,a.cores)
