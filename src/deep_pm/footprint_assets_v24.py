"""Acquire published Stripe 82 HLC and all-sky NSC exposure metadata on Slurm."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import json
import os
from pathlib import Path
import subprocess

from astropy.io import fits
import pandas as pd

from . import access
from .io import digest, save_json

HLC = "https://vizier.cfa.harvard.edu/ftp/cats/V/141"
COUNTS = {0: 446842, 1: 426892, 2: 442984, 3: 128785, 20: 390491,
          21: 844981, 22: 549637, 23: 469936}


def download(url, path):
    if path.is_file():
        return
    partial = path.with_suffix(path.suffix + ".part")
    subprocess.run(["curl", "-fsSL", "--connect-timeout", "15", "--speed-time", "60",
                    "--speed-limit", "1024", "--retry", "3", "-C", "-", "-o",
                    str(partial), url], check=True)
    partial.replace(path)


def hlc_shard(root, hour):
    name = f"HLC.RA_{hour:02d}_to_{hour+1:02d}.fits.gz"
    path = root / name
    receipt = root / (name + ".json")
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        if digest(path) == saved["sha256"] and saved["rows"] == COUNTS[hour]:
            return saved
        raise ValueError("HLC shard or row receipt changed")
    download(f"{HLC}/hlc/{name}", path)
    with gzip.open(path, "rb") as stream:
        # Read to EOF for gzip CRC and length, independent of FITS header.
        while stream.read(4 << 20):
            pass
    with fits.open(path, memmap=False) as hdus:
        table = next(h for h in hdus if isinstance(h, fits.BinTableHDU))
        rows = table.header["NAXIS2"]
        columns = {c.name: dict(unit=c.unit, format=c.format) for c in table.columns}
        if rows != COUNTS[hour] or "RA_PM" not in columns:
            raise ValueError(f"HLC schema/count mismatch for {name}")
    result = dict(rows=rows, bytes=path.stat().st_size, sha256=digest(path), columns=columns)
    save_json(receipt, result)
    print("verified HLC", hour, rows, flush=True)
    return result


def hlc(root, workers):
    download(HLC + "/ReadMe", root / "ReadMe")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        parts = list(pool.map(lambda h: hlc_shard(root, h), COUNTS))
    if any(p["columns"] != parts[0]["columns"] for p in parts):
        raise ValueError("Inconsistent HLC columns across hours")
    save_json(root / "v24_hlc_inventory_summary.json",
              dict(product="V/141 original full HLC FITS", rows=sum(p["rows"] for p in parts),
                   readme_sha256=digest(root / "ReadMe"), columns=parts[0]["columns"],
                   parts={str(h): {k:v for k,v in p.items() if k != "columns"}
                          for h,p in zip(COUNTS, parts)}))


def exposures(root, workers):
    cache = access.Cache(root / "exposure-query")
    total = int(cache.query("nsc", "SELECT COUNT(*) AS n FROM nsc_dr2.exposure",
                            expected=1).iloc[0, 0])
    cols = "exposure,ra,dec,mjd,instrument,filter"
    def shard(lo):
        return access.checked_spatial(cache, "nsc", cols, "nsc_dr2.exposure",
                                      f"ra >= {lo} AND ra < {lo+30}", -90., 90.000001)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        result = pd.concat(pool.map(shard, range(0, 360, 30)), ignore_index=True)
    if len(result) != total or result.exposure.isna().any() or result.exposure.duplicated().any():
        raise ValueError("All-sky exposure partitions disagree with total/unique identifiers")
    path = root / "private_exposures.parquet"
    result.to_parquet(path, index=False)
    save_json(root / "v24_exposure_inventory_summary.json",
              dict(rows=total, sha256=digest(path), instruments=result.instrument.value_counts().to_dict(),
                   columns=cols))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["hlc", "exposures"])
    parser.add_argument("root", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if not os.getenv("SLURM_JOB_ID") or not 1 <= args.workers <= int(os.environ["SLURM_CPUS_PER_TASK"]):
        parser.error("Run within the assigned Slurm CPUs")
    args.root.mkdir(exist_ok=True, parents=True)
    {"hlc": hlc, "exposures": exposures}[args.mode](args.root, args.workers)
