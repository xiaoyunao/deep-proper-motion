"""Summarize normalized survey visits and their overlap across the two pilots."""
import argparse
import json
from pathlib import Path

import pandas as pd


def coverage(base):
    report = {}
    visits = {}
    columns = ["survey", "exposure_id", "detector_id", "observation_time", "band", "instrument"]
    for name in ("repeat_rich", "repeat_sparse"):
        data = pd.read_csv(base/name/"detections.csv", usecols=columns,
                           dtype={"exposure_id": str}, low_memory=False)
        visits[name] = {}
        report[name] = {}
        for survey, subset in data.groupby("survey"):
            visits[name][survey] = set(subset.exposure_id.unique())
            year_bounds = 2000 + (subset.observation_time.agg(["min", "max"])-51544.5)/365.25
            report[name][survey] = dict(unique_visits=len(visits[name][survey]),
                unique_visit_detector=len(subset.drop_duplicates(["exposure_id", "detector_id"])),
                normalized_records=len(subset), bands=subset.band.value_counts().to_dict(),
                instruments=subset.instrument.value_counts().to_dict(),
                year_bounds=year_bounds.round(2).tolist(),
                approx_day_count=int(subset.observation_time.dropna().astype(int).nunique()))
    shared = {s: len(visits["repeat_rich"][s] & visits["repeat_sparse"][s]) for s in ("SDSS", "NSC")}
    return dict(pilots=report, shared=shared)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path)
    args = parser.parse_args()
    print(json.dumps(coverage(args.inputs), allow_nan=False))


if __name__ == "__main__":
    main()
