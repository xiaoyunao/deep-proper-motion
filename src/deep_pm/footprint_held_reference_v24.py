"""Freeze native-sky corrections for the actual archived, held-out reference detections."""
import argparse
import json
import os
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .astrometry import project, unproject
from .full_reference_run import local_support
from .io import digest, save_json


def prepare(base):
    old = base / "v2.2"
    output = base / "v2.4/predecessor-held"
    output.mkdir(parents=True, exist_ok=True)
    graph_meta = old / "graph/v22_global_graph_summary.json"
    components = [v for v in json.loads(graph_meta.read_text())["components"] if v["status"] == "ready"]
    db = duckdb.connect(str(old / "graph/reference.duckdb"), read_only=True)
    db.execute("SET memory_limit='4GB'")
    db.execute(f"SET threads={min(4,int(os.environ['SLURM_CPUS_PER_TASK']))}")
    db.execute(f"SET temp_directory='{output}/spill'")
    signed = {}
    for comp in components:
        key = comp["component"]
        directory = old / "reference" / f"component-{key:04d}"
        source = directory / "private_c0_unfiltered.npz"
        sha = digest(source)
        report = json.loads((directory/"v22_reference_component_summary.json").read_text())
        if sha != report["graph_sha256"]:
            raise ValueError("Changed archived reference solution")
        file = output / f"held-{key:04d}.parquet"
        marker = file.with_suffix(".json")
        ticket = dict(graph=sha, component=key, archive_graph=digest(graph_meta),
                      implementation=digest(Path(__file__)))
        if marker.is_file():
            receipt=json.loads(marker.read_text())
            if receipt["ticket"] != ticket or receipt["sha256"] != digest(file):
                raise ValueError("Changed signed held-out comparison input")
        else:
            db.execute("""CREATE TEMP TABLE IF NOT EXISTS fixed_labels AS
                 SELECT reference FROM (SELECT DISTINCT reference FROM clean WHERE is_holdout)
                 WHERE substr(sha256('v22-held-readout:'||reference),1,2) IN ('00','01','02','03')""")
            f=db.execute("""SELECT c.reference,c.detection_id,c."group",ra_deg,dec_deg FROM clean c
                     SEMI JOIN fixed_labels USING(reference) JOIN group_map USING("group")
                     WHERE is_holdout AND component=? ORDER BY detection_id""",[key]).fetchdf()
            if f.detection_id.duplicated().any():
                raise ValueError("Repeated archived physical detection")
            with np.load(source, allow_pickle=False) as a:
                names=a["names"].astype(str)
                ix=pd.Index(names).get_indexer(f["group"])
                if np.any(ix<0):
                    raise ValueError("Archived group membership differs from the solution")
                center=a["projection_center_deg"]
                supported, _=local_support(f.ra_deg,f.dec_deg,center,f["group"],names,a["draws"])
                xy=project(f.ra_deg,f.dec_deg,center)-a["position"][ix]
                sky=unproject(xy,center)
                f["old_ra"], f["old_dec"] = sky.T
                f["old_supported"] = supported
            pending=file.with_suffix(".pending.parquet")
            f.to_parquet(pending,index=False,compression="zstd")
            pending.replace(file)
            receipt=dict(ticket=ticket, sha256=digest(file), rows=len(f),
                         supported=int(supported.sum()), references=f.reference.nunique())
            save_json(marker,receipt)
        signed[str(key)] = receipt
    db.close()
    save_json(output/"v24_previous_held_reference_summary.json",dict(components=signed,
              rows=sum(v["rows"] for v in signed.values()),supported=sum(v["supported"] for v in signed.values()),
              scope="Actual archived holdout detection identities, old graph offset subtracted in its original chart, then transformed back to ICRS. Native two-axis 12 mas support evaluated separately."))


def archived_positions(base, frame, new_center):
    """Re-express previous corrected directions in the new chart on exact detections."""
    output=base/"v2.4/predecessor-held"
    summary=json.loads((output/"v24_previous_held_reference_summary.json").read_text())
    files=[]
    for key,record in summary["components"].items():
        path=output/f"held-{int(key):04d}.parquet"
        if digest(path)!=record["sha256"]:
            raise ValueError("Changed archived ICRS holdout sample")
        files.append(path)
    previous=pd.concat([pd.read_parquet(p) for p in files],ignore_index=True)
    joined=frame[["reference","detection_id","group","ra_deg","dec_deg"]].merge(
        previous,how="left",on=["reference","detection_id","group"],suffixes=("","_archive"),validate="one_to_one",sort=False)
    present=joined.old_supported.notna().to_numpy()
    for axis in ("ra_deg","dec_deg"):
        np.testing.assert_allclose(joined.loc[present,axis],joined.loc[present,axis+"_archive"],atol=1e-8,rtol=0)
    keep=present & joined.old_supported.eq(True).to_numpy()
    xy=np.full((len(frame),2),np.nan)
    if np.any(keep):
        xy[keep]=project(joined.old_ra[keep],joined.old_dec[keep],new_center)
    return xy,keep,int(present.sum()),digest(output/"v24_previous_held_reference_summary.json")


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("base",type=Path)
    a=p.parse_args()
    if not os.getenv("SLURM_JOB_ID"):
        p.error("Use a Slurm CPU allocation")
    prepare(a.base)
