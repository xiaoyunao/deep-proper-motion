"""Measure held detector overlaps in the continuous C0 reference frame."""
import argparse
import os
from pathlib import Path
from time import monotonic

import numpy as np
import pandas as pd

from .__main__ import tomllib
from .continuous_reference import CENTER, FILES, PRECISION, SPATIAL, bind, read_offsets, verify_directory
from .exposure_catalogues import inside
from .exposure_model import observation_keys
from .io import digest, save_json
from .pilot_production import quality_detections
from .provenance import run_stage
from .public_summary import check_aggregate
from .validation_audit import boundary_pairs, sdss_overlap_rows


def summarize_fields(pairs, stats):
    # The common C0 run/camcol offset cancels for two fields in the same scan.
    pairs = pairs.loc[~pairs.counterpart.duplicated(keep=False)]
    valid_errors = pairs[["err_e", "err_n"]].to_numpy(dtype=float)
    pairs = pairs.loc[np.isfinite(valid_errors).all(axis=1) & (valid_errors > 0).all(axis=1)]
    result = dict(stats, unambiguous_pairs=len(pairs), held_references=pairs.ref.nunique(),
                  held_runs=pairs.run.nunique())
    if len(pairs):
        delta = pairs[["east", "north"]].to_numpy()
        result.update(median_mas=np.repeat(np.median(delta, axis=0)[None, :], 2, axis=0),
                      radial_p68_mas=np.repeat(np.quantile(np.linalg.norm(delta, axis=1), .68), 2))
    return result


def audit(source, reference, graph, sas, spec):
    start = monotonic()
    config = tomllib.loads(PRECISION.read_text())
    prod = tomllib.loads(SPATIAL.read_text())["production"]
    data, xy, _, _, _ = quality_detections(source, CENTER, config, prod)
    keys = observation_keys(data)
    # Old raw-input receipts predate the explicit field column.
    field_ids = pd.Series(keys[1], index=data.index).str.split("|", regex=False).str[1]
    data = data.assign(field_id=pd.to_numeric(field_ids, errors="coerce"))
    matched = pd.read_csv(reference, dtype={"reference": str, "detection_id": str},
                          usecols=["detection_id", "reference", "is_holdout"]).set_index("detection_id")
    refs = matched.reindex(data.detection_id).set_axis(data.index)
    refs = refs.loc[refs.is_holdout.eq(True) & inside(data.ra_deg, data.dec_deg, spec)]
    offsets = read_offsets(graph)
    labels = pd.read_csv(source/"external_labels.csv", dtype=str).set_index("detection_id").original_object_id
    corners = np.array([(x, y) for x in spec["ra"] for y in spec["dec"]])
    # The reference coordinates use the common tangent plane. The SAS query
    # must enclose this core, including real neighbouring SDSS fields.
    radius = np.max(np.linalg.norm(corners-np.asarray(CENTER), axis=1))+.3
    pairs, stats = sdss_overlap_rows(sas, refs, data, xy, keys, (offsets,), CENTER, holdout=True,
                                    search_radius_deg=radius, require_fine_support=False)
    chips = boundary_pairs(refs, data, xy, data.detection_id.map(labels), keys[:1], (offsets,),
                           CENTER, spec, include_geometric=False)
    result = dict(held_core_detections=len(refs), reference_graph_sha256=digest(graph),
                  sdss_adjacent_field=summarize_fields(pairs, stats),
                  nsc_same_exposure_ccd=chips.get("nsc_same_exposure_ccd", {"pair_count": 0}),
                  elapsed_seconds=monotonic()-start)
    check_aggregate(result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("work", "config", "source", "reference", "graph", "sas"):
        p.add_argument(name, type=Path)
    p.add_argument("core")
    a = p.parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        p.error("Run on a Slurm compute node")
    a.work.mkdir(parents=True, exist_ok=True)
    bind(a.work, "source_receipt.json", verify_directory(a.source, FILES))
    bind(a.work, "reference_receipt.json", verify_directory(a.reference, ("reference.csv.gz",)))
    bind(a.work, "graph_receipt.json", verify_directory(a.graph, ("private_c0.npz",)))
    output = "detector_summary.json"
    with run_stage(a.work, a.config, a.core, "continuous-detectors-v12",
                   inputs=("source_receipt.json", "reference_receipt.json", "graph_receipt.json"),
                   outputs=(output,)) as go:
        if go:
            spec = tomllib.loads(a.config.read_text())["fields"][a.core]
            save_json(a.work/output, audit(a.source, a.reference/"reference.csv.gz",
                                           a.graph/"private_c0.npz", a.sas, spec))


if __name__ == "__main__":
    main()
