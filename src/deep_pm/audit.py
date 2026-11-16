"""Compare NSC access routes over a fixed set of exposures."""
import argparse
from pathlib import Path

import pandas as pd

from . import access
from .__main__ import in_cone
from .io import save_json


def run(work, center, radius, margin, n_exposures):
    cache = access.Cache(work / "cache")
    chips = access.covering_chips(cache, *center, radius, margin)
    exposures = chips[["exposure", "mjd"]].drop_duplicates("exposure").sort_values("mjd")
    if exposures.empty:
        raise RuntimeError("No CCD footprints cover the probe")
    exposures = exposures.iloc[[i * (len(exposures) - 1) // max(n_exposures - 1, 1)
                                 for i in range(min(n_exposures, len(exposures)))]]
    chips = chips[chips.exposure.isin(exposures.exposure)]
    new = in_cone(access.fetch_exposure_index(cache, chips, *center, radius), center, radius)
    old = pd.read_csv(work / "nsc_object.csv", usecols=["measid", "exposure", "ra", "dec"])
    old = in_cone(old[old.exposure.isin(exposures.exposure)], center, radius)
    left, right = set(old.measid), set(new.measid)
    summary = dict(exposures=len(exposures), ccds=len(chips), radius_deg=radius,
                   shared=len(left & right), object_only=len(left - right),
                   exposure_only=len(right - left))
    save_json(work / "route_audit_summary.json", summary)
    save_json(work / "route_audit_detail.json", dict(summary, exposures=exposures.to_dict("records")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    parser.add_argument("--ra", type=float, required=True)
    parser.add_argument("--dec", type=float, required=True)
    parser.add_argument("--radius", type=float, required=True)
    parser.add_argument("--count", type=int, default=8)
    args = parser.parse_args()
    run(args.work, (args.ra, args.dec), args.radius, 0.20, args.count)
