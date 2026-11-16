"""Inspect slicing options for the published Xiao2025 binary table."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re

import healpy as hp
import numpy as np
import requests
from requests.packages.urllib3.exceptions import HTTPError

from .astrometry import project
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
        except (requests.RequestException, HTTPError, OSError):
            if attempt == 2:
                raise


def order_probe(schema, n=9):
    batch = 64
    indices = np.linspace(0, schema["catalogue_size"]-batch, n, dtype=int)

    def fetch(index):
        try:
            return get_records(schema, int(index), batch)
        except (requests.RequestException, HTTPError, OSError, ValueError) as exc:
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


def ring(rows):
    return hp.ang2pix(128, rows["RA"].astype(float), rows["DEC"].astype(float), lonlat=True)


def find_ring_boundary(total, pixel, read):
    lo, hi = 0, total
    while lo < hi:
        mid = (lo + hi) // 2
        start = max(lo, mid - 31)
        values = ring(read(start, min(64, hi - start)))
        if np.any(np.diff(values) < 0):
            raise ValueError("FITS rows are locally inconsistent with sorted HEALPix RING")
        index = int(np.searchsorted(values, pixel))
        if index:
            lo = start + index
        if index < len(values):
            hi = start + index
    bracket = ring(read(max(lo - 1, 0), min(total, lo + 1) - max(lo - 1, 0)))
    if (lo > 0 and bracket[0] >= pixel) or (lo < total and bracket[-1] < pixel):
        raise ValueError("Rows bordering the candidate insertion point failed verification")
    return lo


def remote_ground_cutout(work, schema, center, radius_deg, max_rows):
    dtype = fits_dtype(schema)
    total = schema["catalogue_size"]
    chunks = work / "range_cache"
    chunks.mkdir(exist_ok=True)

    def read(start, count):
        path = chunks / f"{start}-{count}.bin"
        if path.exists():
            data = path.read_bytes()
            if len(data) != count*dtype.itemsize:
                raise ValueError("Cached range has inconsistent length")
            return np.frombuffer(data, dtype=dtype)
        batch = get_records(schema, start, count)
        draft = path.with_suffix(".part")
        draft.write_bytes(batch.tobytes())
        draft.replace(path)
        return batch

    pixels = hp.query_disc(128, hp.ang2vec(*center, lonlat=True), np.deg2rad(radius_deg),
                           inclusive=True)
    intervals = list(zip(sorted(pixels), sorted(pixels+1)))
    selected = []
    outcomes = Counter()
    for a, b in intervals:
        try:
            start, stop = [find_ring_boundary(total, edge, read) for edge in (a,b)]
            if not 0 <= stop-start <= max_rows:
                raise ValueError("Pixel exceeds the configured bounded read budget")
            for lo in range(start, stop, 1024):
                records = read(lo, min(1024, stop-lo))
                if np.any(ring(records) != a):
                    raise ValueError("Rows inside the requested pixel contradict the sorted index")
                offset = project(records["RA"], records["DEC"], center)
                good = np.linalg.norm(offset, axis=1) <= np.tan(np.deg2rad(radius_deg))*180/np.pi*3_600_000
                selected.append(records[good & (records["PM_SOURCE"] == b"DESI")])
            outcomes["pixels_verified"] += 1
        except (requests.RequestException, HTTPError, OSError, ValueError) as exc:
            outcomes[type(exc).__name__] += 1
        save_json(work / "legacy_cutout_progress.json",
                  dict(pixels_requested=len(pixels), outcomes=dict(outcomes),
                       verified_http_ranges=len(list(chunks.glob("*.bin")))))
    if outcomes["pixels_verified"] != len(pixels):
        return None
    return np.concatenate(selected) if selected else np.empty(0, dtype=dtype)


def local_ground_cutout(schema, source, center, radius_deg, chunk_rows=500_000):
    """Scan only the two position fields and source flag before copying a small cutout."""
    if source.stat().st_size != schema["file_bytes"]:
        raise ValueError("Catalogue size differs from the verified FITS product")
    table = np.memmap(source, mode="r", dtype=fits_dtype(schema),
                      offset=schema["table_data_offset"], shape=(schema["catalogue_size"],))
    selected = []
    for start in range(0, len(table), chunk_rows):
        chunk = table[start:start+chunk_rows]
        dra = (chunk["RA"] - center[0] + 180) % 360 - 180
        box = (np.abs(dra) < radius_deg/np.cos(np.deg2rad(abs(center[1])+radius_deg)))
        box &= np.abs(chunk["DEC"] - center[1]) < radius_deg
        box &= chunk["PM_SOURCE"] == b"DESI"
        filtered = chunk[box]
        pos = project(filtered["RA"], filtered["DEC"], center)
        selected.append(filtered[np.linalg.norm(pos, axis=1) <= radius_deg * 3_600_000])
    return np.concatenate(selected)



def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("work", type=Path)
    p.add_argument("--ranged-cutout", action="store_true",
                   help="try a bounded HEALPix RING range extraction with per-range checkpointing")
    p.add_argument("--max-pixel-rows", default=20_000, type=int)
    p.add_argument("--local-fits", type=Path,
                   help="verified full FITS on remote storage; export only a Scratch subset")
    p.add_argument("--region", default="test_equator")
    p.add_argument("--config", type=Path, default=Path("configs/precision-v03.toml"))
    args = p.parse_args()
    schema = json.loads((args.work / "legacy_header_summary.json").read_text())
    if args.local_fits is not None or args.ranged_cutout:
        if args.local_fits is not None and args.ranged_cutout:
            p.error("Select either a verified local copy or a remote ranged cutout.")
        from .__main__ import tomllib

        if not os.environ.get("SLURM_JOB_ID"):
            p.error("Catalogue extraction requires a Slurm compute node.")
        region = tomllib.loads(args.config.read_text())["regions"][args.region]
        center = (region["ra"], region["dec"])
        rows = (remote_ground_cutout(args.work, schema, center, region["radius"], args.max_pixel_rows)
                if args.ranged_cutout else local_ground_cutout(schema, args.local_fits, center, region["radius"]))
        if rows is None:
            p.exit(1, "The requested cutout has incomplete pixel coverage; consult the aggregate progress file.\n")
        output = args.work / ("legacy_ground_" + args.region + ".npy")
        if output.exists():
            p.error("The cutout path already exists. Use a separate run directory.")
        with output.open("xb") as handle:
            np.save(handle, rows, allow_pickle=False)
        save_json(args.work / ("legacy_ground_" + args.region + "_count.json"), dict(ground_rows=len(rows)))
        return
    with run_stage(args.work, args.config, args.region, "legacy-order-v03",
                   inputs=["legacy_header_summary.json"], outputs=["legacy_order_summary.json"]) as run:
        if run:
            save_json(args.work / "legacy_order_summary.json", order_probe(schema))


if __name__ == "__main__":
    main()
