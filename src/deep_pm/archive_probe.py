"""Count-check two small COSMOS download shards and verify cache replay."""
import argparse
from collections import Counter
from pathlib import Path
import os
from time import monotonic

import numpy as np

from . import access
from .io import save_json
from .provenance import run_stage


class MeteredCache(access.Cache):
    def __init__(self, path):
        super().__init__(path)
        self.calls = Counter()

    def query(self, service, query, expected=None):
        self.calls["checked_queries"] += 1
        return super().query(service, query, expected)


def probe(work):
    # Two disjoint caps inside the v0.5 COSMOS core. No truncation or sampling
    # within a cap, and no broader acquisition if the bounded cap is oversized.
    a, d = 150.12, 2.20
    radius = .012
    bound = 750
    definitions = [(a-.018,d),(a+.018,d)]
    result = {}
    for label,(ra,dec) in zip(("west","east"),definitions):
        cache = MeteredCache(work/"cache"/label)
        params = access.q3c(ra,dec,radius)+" AND ndetr>0 AND (rmag<24 OR nphotr=0 OR rmag IS NULL)"
        batches = []
        snapshots = []
        for phase in ("initial_or_resumed","cache_replay"):
            seconds = monotonic()
            start = cache.calls["checked_queries"]
            try:
                index = cache.checked("nsc","id AS objectid","nsc_dr2.object",params,cap=bound)
            except access.PartitionTooLarge:
                result[label] = {"status":"index_over_bound"}
                break
            names = index.objectid.astype(str).sort_values()
            data = access.fetch_measurements_for_objects(cache,names,batches.append,batch_size=100)
            snapshots.append((set(names),set(data.measid.astype(str))))
            if not set(data.objectid.astype(str)) <= set(names) or data.measid.duplicated().any():
                raise ValueError("The archive returned duplicate measurements or out-of-shard aliases")
            result.setdefault(label,{})[phase] = dict(index_objects=len(index), measurements=len(data),
                    objects_with_measurements=data.objectid.nunique(), checked_queries=cache.calls["checked_queries"]-start,
                    elapsed_seconds=monotonic()-seconds)
        if len(snapshots)==2:
            if snapshots[0]!=snapshots[1]:
                raise ValueError("Resumed query did not reproduce the returned detection set")
            result[label]["status"] = "complete"
    area = 4*np.pi*(1-np.cos(np.deg2rad(radius)))*(180/np.pi)**2
    return dict(area_sqdeg=area, per_shard_object_cap=bound, shards=result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work",type=Path)
    parser.add_argument("--config",type=Path,default=Path("configs/astrometry-v06.toml"))
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("Use a Slurm compute node.")
    args.work.mkdir(parents=True,exist_ok=True)
    with run_stage(args.work,args.config,"cosmos_probe","archive-probe-v06",
                   outputs=["archive_probe_summary.json"]) as go:
        if go:
            save_json(args.work/"archive_probe_summary.json",probe(args.work))


if __name__=="__main__":
    main()
