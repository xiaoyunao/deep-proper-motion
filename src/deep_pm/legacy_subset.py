"""Inspect slicing options for the published Xiao2025 binary table."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re

import healpy as hp
import numpy as np
import requests

from .io import save_json
from .precision_inputs import XIAO_URL
from .provenance import run_stage


def fits_dtype(schema):
    kinds = {"K": ">i8", "J": ">i4", "D": ">f8", "E": ">f4", "I": ">i2", "B": "u1"}
    columns = []
    for field, (form, _) in schema["columns"].items():
        match = re.fullmatch(r"([0-9]*)([A-Z])", form)
        count, kind = int(match[1] or 1), match[2]
        if kind == "A":
            columns.append((field, f"S{count}"))
        elif count == 1:
            columns.append((field, kinds[kind]))
        else:
            raise ValueError("Non-scalar numeric FITS format was not expected")
    dtype = np.dtype(columns)
    if dtype.itemsize != schema["row_bytes"]:
        raise ValueError("FITS column widths do not match the verified row size")
    return dtype


def get_records(schema, start, count, uri=XIAO_URL):
    dtype = fits_dtype(schema)
    lo = schema["table_data_offset"] + start*dtype.itemsize
    hi = lo + count*dtype.itemsize - 1
    expect = f"bytes {lo}-{hi}/{schema['file_bytes']}"
    for attempt in range(3):
        try:
            with requests.get(uri, stream=True, headers={"Range": f"bytes={lo}-{hi}",
                              "Accept-Encoding": "identity"}, timeout=(12, 20)) as response:
                if response.status_code != 206 or response.headers.get("Content-Range") != expect:
                    raise ValueError("Catalogue server did not return the requested byte range")
                content = response.raw.read(hi-lo+1)
            if len(content) != hi-lo+1:
                raise ValueError("Catalogue range was truncated")
            return np.frombuffer(content, dtype=dtype)
        except (requests.RequestException, OSError):
            if attempt == 2:
                raise


def order_probe(schema, n=9):
    batch = 64
    indices = np.linspace(0, schema["catalogue_size"]-batch, n, dtype=int)

    def fetch(index):
        try:
            return get_records(schema, int(index), batch)
        except (requests.RequestException, OSError, ValueError) as exc:
            return type(exc).__name__

    with ThreadPoolExecutor(max_workers=2) as pool:
        fetched = list(pool.map(fetch, indices))
    chunks = [chunk for chunk in fetched if not isinstance(chunk, str)]
    result = dict(ranges_requested=n, ranges_read=len(chunks),
                  failed_ranges=sorted([item for item in fetched if isinstance(item, str)]))
    if not chunks:
        return dict(result, sampled_records=0)
    rows = np.concatenate(chunks)
    ra, dec = rows["RA"], rows["DEC"]
    key = {"longitude": ra, "latitude": dec}
    for nested in (False, True):
        pix = hp.ang2pix(128, np.array(ra, dtype=float), np.array(dec, dtype=float),
                         nest=nested, lonlat=True)
        label = "nested" if nested else "ring"
        key[label] = pix
        key[label+"_as_text"] = pix.astype(str)
    counts = {name: int(np.count_nonzero(data[1:] < data[:-1])) for name, data in key.items()}
    labels, c = np.unique(np.char.strip(rows["PM_SOURCE"].astype(str)), return_counts=True)
    return dict(result, sampled_records=len(rows), adjacent_decreases=counts,
                source_flag_counts={label: int(count) for label, count in zip(labels, c)})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("work", type=Path)
    p.add_argument("--config", type=Path, default=Path("configs/precision-v03.toml"))
    args = p.parse_args()
    schema = json.loads((args.work / "legacy_header_summary.json").read_text())
    with run_stage(args.work, args.config, "test_equator", "legacy-order-v03",
                   inputs=["legacy_header_summary.json"], outputs=["legacy_order_summary.json"]) as run:
        if run:
            save_json(args.work / "legacy_order_summary.json", order_probe(schema))


if __name__ == "__main__":
    main()
