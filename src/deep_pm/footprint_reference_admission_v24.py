"""Check the joint physical solution, gauge, and synchronized draws before source fitting."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .io import digest,save_json
from . import stereographic


def run(root,predecessor_output=None):
    registered=digest(root/"registration/private_full_coordinate_plan.json")
    if predecessor_output is None:
        registration=json.loads((root.parent/"registration/v24_graph_registration_summary.json").read_text())
        if registered!=registration["graph_registration_sha256"]:raise ValueError("Changed registered science inputs")
        output=root/"admission/v24_reference_admission_summary.json"
    else:
        if root.resolve() in predecessor_output.resolve().parents:raise ValueError("Preserve the frozen predecessor")
        output=predecessor_output
    meta=root/"graph/v22_global_graph_summary.json"
    mapping=pd.read_parquet(root/"graph/private_group_components.parquet")
    if mapping["group"].duplicated().any():raise ValueError("One physical group has contradictory reference components")
    components=json.loads(meta.read_text())["components"]
    proofs={}
    groups=0
    for component in components:
        if component["status"]!="ready":continue
        key=component["component"]
        directory=root/"reference"/f"component-{key:04d}"
        path=directory/"private_c0_unfiltered.npz"
        summary=json.loads((directory/"v22_reference_component_summary.json").read_text())
        signature=digest(path)
        if signature!=summary["graph_sha256"] or summary["component"]!=key:raise ValueError("Reference file changed")
        expected=set(mapping.loc[mapping.component==key,"group"])
        with np.load(path,allow_pickle=False) as archive:
            names=archive["names"].astype(str)
            count=len(names)
            if len(expected)!=count or set(names)!=expected:raise ValueError("Inconsistent physical group mapping")
            point=archive["position"]
            draws=archive["draws"]
            if point.shape!=(count,2) or draws.shape!=(count,48,2):raise ValueError("Invalid point or draw axes")
            if not np.isfinite(point).all() or not np.isfinite(draws).all():raise ValueError("Invalid reference state")
            nsc=np.char.startswith(names,"NSC:")
            if not np.any(nsc) or not np.any(~nsc) or np.any(archive["counts"]<10):raise ValueError("Invalid connected reference support")
            np.testing.assert_allclose(np.median(point[nsc],axis=0),0,atol=1e-9,rtol=0)
            np.testing.assert_allclose(np.median(draws[nsc],axis=0),0,atol=1e-9,rtol=0)
            chart=stereographic.from_record(archive)
            if stereographic.kind(chart)!=component.get("projection_kind","gnomonic"):
                raise ValueError("Reference archive and graph disagree on the coordinate representation")
            theta=component["maximum_theta_deg"]
            if isinstance(chart,stereographic.Stereographic):
                proof=root.parent/"registration/v24_stereographic_replay_summary.json"
                if not stereographic.admitted(theta) or digest(proof)!=component["wide_chart_qualification_sha256"]:
                    raise ValueError("Unqualified wide reference projection")
            elif not 0<=theta<85:
                raise ValueError("Unverified gnomonic projection chart")
            np.testing.assert_allclose(archive["projection_center_deg"],component["center_deg"],rtol=0,atol=1e-12)
            centered=draws-draws.mean(axis=1,keepdims=True)
            covariance=centered.transpose(0,2,1)@centered/47
            eig=np.linalg.eigvalsh(covariance)
            if not np.isfinite(eig).all() or np.any(eig[:,0]<-1e-10*np.maximum(eig[:,1],1)):
                raise ValueError("Reference group covariance is inconsistent")
        proofs[str(key)]=signature
        groups+=count
    if not proofs:raise ValueError("No cross-survey physical reference solution")
    save_json(output,dict(predecessor_check=predecessor_output is not None,graph_registration_sha256=registered,
              graph_metadata_sha256=digest(meta),solved_components=len(proofs),solved_groups=groups,
              group_map_sha256=digest(root/"graph/private_group_components.parquet"),graphs=proofs,
              consistent_physical_groups_and_covariance=True,draws=48,
              scope="Physical graph/gauge/draw integrity. Spatial zero points and stellar population accuracy require separate validation."))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("root",type=Path);p.add_argument("--predecessor-output",type=Path);a=p.parse_args()
    if not os.getenv("SLURM_JOB_ID"):p.error("Use assigned Slurm CPUs")
    run(a.root,a.predecessor_output)
