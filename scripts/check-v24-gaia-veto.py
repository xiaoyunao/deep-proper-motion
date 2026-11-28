"""Replay the frozen galaxy selection with an independently obtained Gaia position index."""
import argparse
import json
import os
from pathlib import Path

from deep_pm.continuous_reference import verify_directory
from deep_pm.full_reference_input import extract_one
from deep_pm.io import digest, save_json
from deep_pm.provenance import run_stage


def run(work, name, preregistration):
    reg = work/"registration"
    trial = work/"gaia-veto-trial"
    comparison = reg/"v24_gaia_veto_controls_summary.json"
    checked = json.loads(comparison.read_text())
    holdout = json.loads(preregistration.read_text())
    if (holdout["core"] != name or holdout["footprint"] != checked["footprint"]
            or len(checked["controls"]) != 5 or not checked["position_membership_passes"]):
        raise ValueError("Position membership or independent field registration failed")
    original = work/"intake"/name
    old_shard = work/"input"/name
    resolved = work/"resolved"/(name+".json")
    if resolved.exists():
        pointers=json.loads(resolved.read_text())
        original,old_shard=Path(pointers["source"]),Path(pointers["shard"])
    frozen=["detections.csv","external_labels.csv","spectra_veto.csv"]
    original_manifest=verify_directory(original,frozen+["validation/gaia.csv"])
    sky=trial/name/"gaia.csv"
    provenance=json.loads((trial/name/"private_provenance.json").read_text())
    if provenance["controls_sha256"]!=digest(comparison) or provenance["gaia_sha256"]!=digest(sky):
        raise ValueError("Different external positions from the control comparison")
    local=trial/"fixed-input"/name
    (local/"validation").mkdir(parents=True,exist_ok=True)
    config=local/"registered.toml"
    text=(f'footprint = "{checked["footprint"]}"\noriginal_manifest = "{digest(original_manifest)}"\n'
          f'gaia_source_subset = "{digest(sky)}"\npreregistration = "{digest(preregistration)}"\n')
    if config.exists() and config.read_text()!=text:
        raise ValueError("Different fixed-input replay")
    config.write_text(text)
    for file,src in [(k,original/k) for k in frozen]+[("validation/gaia.csv",sky)]:
        dest=local/file
        if dest.exists() and dest.resolve()!=src.resolve():
            raise ValueError("Different original scientific input")
        if not dest.exists():
            dest.symlink_to(src.resolve())
    with run_stage(local,config,name,"fixed-detections-alternate-gaia",outputs=frozen+["validation/gaia.csv"]):
        pass
    spec=json.loads((reg/"private_full_footprint_plan.json").read_text())["new"][name]
    entry=dict(core=name,source=str(local),source_receipt_sha256=digest(local/"run_manifest.json"),
               center=[sum(spec[k])/2 for k in ("ra","dec")])
    now=extract_one((entry,trial/"reference"))
    old=json.loads((old_shard/"v22_reference_input_core_summary.json").read_text())
    old_sha=digest(old_shard/"reference.parquet")
    if old_sha!=old["parquet_sha256"]:
        raise ValueError("Frozen reference shard changed")
    same=now["parquet_sha256"]==old_sha
    save_json(reg/"v24_independent_gaia_veto_summary.json",dict(footprint=checked["footprint"],
              slurm_job=os.environ["SLURM_JOB_ID"],selection_sha256=digest(preregistration),
              controls_sha256=digest(comparison),fixed_detection_rows=now["input_detections"],
              reference_rows=now["measurements"],reference_file_byte_equal=same,
              original_sha256=old_sha,replayed_sha256=now["parquet_sha256"],
              interpretation="Same normalized detections and original frozen reference implementation. Scope is the galaxy position veto only."))
    if not same:
        raise ValueError("Alternative veto changed the actual reference")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work",type=Path)
    parser.add_argument("core")
    parser.add_argument("preregistration",type=Path)
    args=parser.parse_args()
    if not os.getenv("SLURM_JOB_ID"):
        parser.error("Run in a Slurm allocation")
    run(args.work,args.core,args.preregistration)
