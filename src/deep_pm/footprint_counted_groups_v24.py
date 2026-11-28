"""Check original measurement pages against a separate archive count for every physical object."""
from hashlib import sha256
import json

import numpy as np
import pandas as pd

from . import access
from .io import save_json


def query_key(sql):
    return sha256(("nsc\n" + sql).encode()).hexdigest()


def strips(archive, south, north):
    """Split the independent archive query before its object or measurement load gets large."""
    todo = [(south, north, archive)]
    while todo:
        low, high, sample = todo.pop()
        if len(sample) <= 60000 and sample.ndet.sum() <= 1200000:
            yield low, high
            continue
        mid = round((low + high) / 2, 9)
        if mid <= low or mid >= high:
            raise ValueError("Cannot isolate a bounded archive declination strip")
        mask = sample.dec < mid
        todo.extend([(mid, high, sample.loc[~mask]), (low, mid, sample.loc[mask])])


class GroupCounts:
    def __init__(self, counts):
        if counts.index.has_duplicates or counts.index.hasnans:
            raise ValueError("An object appears in overlapping archive count strips")
        self.counts = counts
        self.seen = np.zeros(len(counts), dtype=bool)
        self.pages = 0
        self.measurements = 0

    def check(self, page):
        if page.objectid.isna().any() or page.measid.isna().any() or page.measid.duplicated().any():
            raise ValueError("Missing or repeated physical measurement identity")
        histogram = page.objectid.value_counts()
        index = self.counts.index.get_indexer(histogram.index)
        if np.any(index < 0) or np.any(self.seen[index]):
            raise ValueError("Unknown object or repeated object across measurement pages")
        if not np.array_equal(histogram.to_numpy(), self.counts.to_numpy()[index]):
            raise ValueError("Measurement frequencies disagree with independent archive counts")
        self.seen[index] = True
        self.pages += 1
        self.measurements += len(page)

    def finish(self):
        if not self.seen.all() or self.measurements != self.counts.sum():
            raise ValueError("Missing objects or measurements after complete page traversal")


def independent_counts(cache, archive, table, where, south, north):
    parts = []
    evidence = []
    for low, high in strips(archive[["dec", "ndet"]], south, north):
        clause = f" FROM {table} WHERE {where} AND o.dec>={low:.9f} AND o.dec<{high:.9f}"
        total_sql = "SELECT COUNT(*) AS n" + clause
        group_sql = "SELECT o.id AS objectid, COUNT(*) AS n" + clause + " GROUP BY o.id"
        total = int(cache.query("nsc", total_sql, expected=1).iloc[0, 0])
        frame = cache.query("nsc", group_sql)
        if list(frame) != ["objectid", "n"] or frame.objectid.isna().any():
            raise ValueError("Invalid grouped archive response")
        if len(frame) and (not pd.api.types.is_integer_dtype(frame.n) or (frame.n <= 0).any()):
            raise ValueError("Invalid grouped archive measurement counts")
        if frame.n.sum() != total:
            raise ValueError("Truncated or inconsistent grouped archive result")
        parts.append(frame.set_index("objectid").n.astype("int64"))
        for sql in (total_sql, group_sql):
            key = query_key(sql)
            record = json.loads((cache.path / (key + ".json")).read_text())
            evidence.append(dict(query_key=key, payload_sha256=record["sha256"]))
    result = GroupCounts(pd.concat(parts))
    if not result.counts.index.isin(archive.objectid).all():
        raise ValueError("Measured object absent from the registered spatial archive")
    return result, evidence


def measurement_page(cache, columns, table, where, cap):
    # Preserve already verified native results, including their original query and bytes.
    native = f"SELECT {columns},COUNT(*) OVER () AS verified_partition_count FROM {table} WHERE {where} LIMIT {cap+1}"
    if (cache.path / (query_key(native) + ".json")).is_file():
        return cache.checked_native_window(columns, table, where, cap=cap)

    def bounded(frame):
        if len(frame) > cap:
            raise access.PartitionTooLarge("Measurement page exceeds the registered partition limit")
        return list(frame) == access.MEAS.split(",")

    return cache.query("nsc", f"SELECT {columns} FROM {table} WHERE {where}", validate=bounded)


def save_evidence(cache, counts, evidence):
    counts.finish()
    save_json(cache.path.parent / "private_grouped_measurement_counts.json",
              dict(method="disjoint object counts plus independent strip totals",
                   archive_queries=evidence, measured_objects=len(counts.counts),
                   measurements=counts.measurements, pages=counts.pages))
