"""Audit common-epoch ownership and close cross-core catalogue pairs."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .__main__ import tomllib
from .continuous_reference import verify_directory
from .exposure_catalogues import coordinates, inside
from .io import digest, save_json
from .public_summary import check_aggregate
from .astrometry import project
from .standard_motion import icrs_motion


def close_pairs(data,centers=None):
    tree = cKDTree(coordinates(data.ra, data.dec))
    cores = data.core.to_numpy()
    motion = data[["pm_east", "pm_north"]].to_numpy(dtype=float)
    results = {}
    for radius in (.05, .2):
        chord = 2*np.sin(np.deg2rad(radius/3600)/2)
        pairs = np.asarray(sorted(tree.query_pairs(chord)), dtype=int).reshape(-1, 2)
        cross = cores[pairs[:, 0]] != cores[pairs[:, 1]]
        pairs = pairs[cross]
        native=motion
        if centers is not None:
            native=motion.copy()
            for i in np.unique(pairs):
                row=data.iloc[i];center=centers[row.core]
                point=project([row.ra],[row.dec],center)[0]
                native[i]=icrs_motion(point,motion[i],center)['pm_icrs_masyr']
        separation = np.linalg.norm(native[pairs[:, 0]]-native[pairs[:, 1]], axis=1)
        results[str(radius)] = dict(cross_core_pairs=len(pairs),
                                   motion_difference_below_1_masyr=int((separation < 1).sum()))
    return dict(catalogue_count=len(data), exact_position_duplicate_rows=int(data.duplicated(["ra", "dec"], keep=False).sum()),
                cross_core_close_pairs_by_arcsec=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "config", "graph", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        parser.error("Run on a Slurm compute node")
    specs = tomllib.loads(args.config.read_text())["fields"]
    graph_hash = digest(args.graph/"private_c0.npz")
    frames = []
    receipts = {}
    outside = 0
    for core, directory in json.loads(args.plan.read_text()).items():
        directory = Path(directory)
        receipt = verify_directory(directory, ("continuous_core_summary.json", "private_catalogue.csv.gz"))
        summary = json.loads((directory/"continuous_core_summary.json").read_text())
        if summary["shared_solution_sha256"] != graph_hash:
            raise ValueError(f"Reference graph changed for {core}")
        data = pd.read_csv(directory/"private_catalogue.csv.gz", dtype=float,
                           usecols=["ra", "dec", "pm_east", "pm_north"])
        if len(data) != summary["owned_w0"] or not np.isfinite(data.to_numpy()).all():
            raise ValueError(f"Invalid catalogue coordinates or row count for {core}")
        outside += int((~inside(data.ra, data.dec, specs[core])).sum())
        frames.append(data.assign(core=core))
        receipts[core] = digest(receipt)
    result = close_pairs(pd.concat(frames, ignore_index=True))
    result.update(cores=len(frames), outside_common_epoch_owner=outside, graph_sha256=graph_hash,
                  fit_receipt_sha256=receipts, slurm_job=os.environ["SLURM_JOB_ID"],
                  close_pairs_are_candidates=True)
    check_aggregate(result)
    save_json(args.output, result)
    if outside:
        parser.exit(2, "Sources found outside their common-epoch owner.\n")


if __name__ == "__main__":
    main()
