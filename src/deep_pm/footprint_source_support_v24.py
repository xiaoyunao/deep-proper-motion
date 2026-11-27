"""Aggregate a closed blind catalogue by photometric and temporal support, once."""
import argparse
import json
import os
from pathlib import Path

from .io import digest,save_json


def run(root,output):
    import duckdb
    import numpy as np
    from .full_coordinate_production_readout import aggregate
    closure=root/"catalogue/v22_catalogue_ownership_summary.json"
    plan=root/"registration/private_full_coordinate_plan.json"
    registered=json.loads(plan.read_text())
    closed=json.loads(closure.read_text())
    ledger=root/"catalogue/private_closure_binding.json"
    binding=json.loads(ledger.read_text())
    if binding["plan_sha256"]!=digest(plan) or digest(ledger)!=closed["private_input_binding_sha256"]:
        raise ValueError("Final source index and registration disagree")
    db=duckdb.connect(str(root/"catalogue/catalogue.duckdb"),read_only=True)
    workers=min(8,int(os.getenv("SLURM_CPUS_PER_TASK",1)))
    db.execute(f"SET threads={workers}")
    db.execute("SET memory_limit='4GB'")
    # The missing-r category remains in closure and the source-count table.
    cells=db.execute("""
        SELECT CASE WHEN magnitude IS NULL OR NOT isfinite(magnitude) THEN NULL
                    ELSE greatest(12,least(25,floor(magnitude*2)/2)) END AS r_low,
               CASE WHEN joint_span>=30 THEN 30 ELSE floor(joint_span/2)*2 END AS year_low,
               sdss_runs=1 AS single_run,kind,count(*) AS n,
               median(se_e) AS se_e,median(se_n) AS se_n,median(nsc_nights) AS nights,
               median(maximum_group_leverage) AS leverage,
               avg(CASE WHEN maximum_group_leverage>=.8 THEN 1. ELSE 0. END) AS high_leverage_fraction
        FROM accepted GROUP BY ALL ORDER BY r_low,year_low,single_run,kind
    """).fetchdf()
    # Panel medians must be computed from the underlying sources, not averaged
    # over the unequal populations stored in the more detailed breakdown.
    panels=db.execute("""
        SELECT floor(magnitude*2)/2 AS r_low,
               CASE WHEN joint_span>=30 THEN 30 ELSE floor(joint_span/2)*2 END AS year_low,
               sdss_runs=1 AS single_run,count(*) AS n,
               median(sqrt(se_e*se_e+se_n*se_n)) AS vector_standard_error,
               median(maximum_group_leverage) AS leverage
        FROM accepted WHERE magnitude>=16 AND magnitude<24 GROUP BY ALL ORDER BY ALL
    """).fetchdf()
    sky=db.execute("""SELECT floor(ra_deg)::INTEGER AS lon,
           least(89,floor(dec_deg)::INTEGER) AS lat,count(*) AS n
           FROM accepted GROUP BY ALL ORDER BY ALL""").fetchall()
    if sum(row[2] for row in sky)!=closed["retained_common_epoch_solutions"]:
        raise ValueError("Sky-cell counts disagree with frozen source ownership")
    db.close()
    total=int(cells.n.sum())
    if total!=closed["retained_common_epoch_solutions"]:raise ValueError("Source bins do not close")
    def pack(frame):
        return json.loads(frame.to_json(orient="records",double_precision=12))
    proof=[]
    for entry in registered["cores"]:
        source=root/"sources"/entry["core"]/"v22_source_core_summary.json"
        proof.append(json.loads(source.read_text()))
    selection=aggregate(proof)
    if not np.isfinite(panels.vector_standard_error).all():raise ValueError("Invalid adopted errors")
    admission=root/"admission/v24_reference_admission_summary.json"
    save_json(output,dict(reference_admission_sha256=digest(admission) if admission.exists() else None,closure_sha256=digest(closure),catalogue_count=total,conditions=pack(cells),figure_bins=pack(panels),sky_source_counts=sky,
              preownership_funnel=selection,scope="Counts of accepted J2015.5 objects and conditional formal errors. "
              "Vector standard error is sqrt(C_EE+C_NN), with component covariance retained in the private catalogue. "
              "Counts do not measure recovery of a photometric parent. Magnitude uses adopted observed r, "
              "outer magnitude bins include overflow; the >=30 yr time bin includes all longer baselines."))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("root",type=Path);p.add_argument("output",type=Path);a=p.parse_args()
    if not os.getenv("SLURM_JOB_ID"):p.error("Use assigned Slurm CPUs")
    run(a.root,a.output)
