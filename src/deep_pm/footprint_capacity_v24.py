"""Count whole-sky CCD keys and detector load without reading individual motions."""
import argparse
import json
import os
from pathlib import Path

import duckdb
import numpy as np

from .access import Cache
from .io import digest,save_json


def run(base):
    work=base/"v2.4"
    output=work/"capacity"
    output.mkdir(exist_ok=True,parents=True)
    chips=work/"chips"
    db=duckdb.connect()
    db.execute("SET memory_limit='3GB'")
    db.execute(f"SET threads={min(4,int(os.environ['SLURM_CPUS_PER_TASK']))}")
    db.execute(f"SET temp_directory='{output}/spill'")
    n,duplicates,nulls=db.execute(f"""SELECT sum(n),sum(n-1),
        sum(CASE WHEN exposure IS NULL OR ccdnum IS NULL THEN n ELSE 0 END)
        FROM (SELECT exposure,ccdnum,count(*) n FROM read_parquet('{chips}/*.parquet')
              GROUP BY exposure,ccdnum) t""").fetchone()
    expected=json.loads((chips/"private_chip_manifest.json").read_text())["total"]
    audit=dict(physical_rows=int(n),duplicate_physical_keys=int(duplicates),missing_physical_keys=int(nulls))
    save_json(output/"v24_physical_ccd_summary.json",audit)
    if n!=expected or duplicates or nulls:
        raise ValueError("Duplicate, missing or uncounted physical CCDs")
    cache=Cache(output/"queries")
    query=("SELECT CASE WHEN ra>=-360 AND ra<720 AND abs(dec)<=90 THEN floor(ra)::int ELSE -9999 END AS a,"
           "CASE WHEN abs(dec)<=90 THEN floor(dec)::int ELSE -9999 END AS d,"
           "count(*) AS chips,sum(nmeas::bigint) AS meas FROM nsc_dr2.chip")
    parts=[]
    # Half-open declination zones bound transfer size; separate invalid rows.
    for south in range(-90,90,10):
        where=f" WHERE dec>={south} AND dec{'<=' if south==80 else '<'}{south+10}"
        parts.append(cache.query("nsc",query+where+" GROUP BY 1,2"))
    parts.append(cache.query("nsc",query+" WHERE (abs(dec)<=90) IS NOT TRUE GROUP BY 1,2"))
    import pandas as pd
    df=pd.concat(parts,ignore_index=True)
    if df.chips.sum()!=expected or df.meas.isna().any() or (df.meas<0).any():
        raise ValueError("All-sky detector load failed census")
    file=output/"private_unit_degree_load.parquet"
    df.to_parquet(file,index=False)
    result=dict(**audit,all_ccd_nmeas=int(df.meas.sum()),load_sha256=digest(file))
    manifest=work/"registration/private_full_footprint_plan.json"
    if manifest.is_file():
        cells=json.loads(manifest.read_text())["new"]
        full=np.zeros((180,360))
        for spec in cells.values():
            j=int(np.floor(spec["ra"][0]))%360;i=int(np.floor(spec["dec"][0]))+90
            full[i,j]+=spec["time_area"]
        mass=0.;envelope=0
        for el in df.itertuples():
            if -90<=el.d<90 and -360<=el.a<720:
                nominal=np.degrees(np.sin(np.radians(el.d+1))-np.sin(np.radians(el.d)))
                weight=min(1.,max(0.,full[el.d+90,el.a%360]/nominal))
                mass+=el.meas*weight;envelope+=int(el.meas)*(weight>0)
        result.update(manifest_sha256=digest(manifest),
                      new_nmeas_area_weighted=int(mass),new_nonempty_degree_centres_nmeas=int(envelope),
                      interpretation="Planning estimate weighted by native footprint area within a degree. CCD-centre load need not be spatially uniform; independent of archive row/quality/candidate cuts.")
    save_json(output/"v24_detector_load_summary.json",result)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("base",type=Path);a=p.parse_args()
    if not os.getenv("SLURM_JOB_ID"):p.error("Use an allocated Slurm CPU node")
    run(a.base)
