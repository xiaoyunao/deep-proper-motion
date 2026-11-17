"""Prepare a frozen rectangular holdout from a complete circular acquisition."""
import argparse
import json
import os
from pathlib import Path

from astropy.io import fits
import numpy as np
import pandas as pd

from .__main__ import tomllib
from .exposure_catalogues import inside
from .io import save_json
from .provenance import run_stage


def legacy_cutout(path, destination, box, halo):
    fields = ("RA", "DEC", "PM_SOURCE", "pmRA", "pmDEC", "pmRA_err", "pmDEC_err")
    pieces = []
    with fits.open(path, memmap=True, character_as_bytes=True) as handle:
        table = handle[1].data
        names = {k.casefold(): k for k in table.names}
        for lo in range(0, len(table), 1_000_000):
            sl = slice(lo, lo+1_000_000)
            ra, dec = (table[names[k]][sl] for k in ("ra", "dec"))
            select = np.flatnonzero(inside(ra, dec, box, halo)) + lo
            if len(select):
                rows = {}
                for field in fields:
                    v = table[names[field.casefold()]][select]
                    rows[field] = (np.char.strip(v.astype(str)) if field == "PM_SOURCE"
                                   else v.astype(v.dtype.newbyteorder("="), copy=False))
                pieces.append(pd.DataFrame(rows))
    result = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=fields)
    result.to_csv(destination, index=False)


def prepare(work, source, archives, legacy, box, halo):
    # The parent receipt and files stay in Scratch. Only this subset is fitted.
    data = pd.read_csv(source/"detections.csv", low_memory=False)
    data = data.loc[inside(data.ra_deg, data.dec_deg, box, halo)]
    data.to_csv(work/"detections.csv", index=False)
    labels = pd.read_csv(source/"external_labels.csv", dtype=str)
    labels.loc[labels.detection_id.isin(data.detection_id)].to_csv(work/"external_labels.csv", index=False)
    archive = work/"official_nsc.csv"
    if not archive.exists():
        archive.symlink_to((source/"official_nsc.csv").resolve(strict=True))
    if archive.resolve() != (source/"official_nsc.csv").resolve(strict=True):
        raise ValueError("NSC archive slice changed")
    gaia = pd.read_csv(source/"validation/gaia.csv")
    gaia = gaia.loc[inside(gaia.ra, gaia.dec, box, halo)]
    (work/"validation").mkdir(exist_ok=True)
    gaia.to_csv(work/"validation/gaia.csv", index=False)
    for stem in ("spectra_union", "spectra_veto"):
        columns = ["ra", "dec", "reference_r", "origin"]
        chunks = []
        for frame in pd.read_csv(archives/f"{stem}.csv", usecols=columns, chunksize=300_000):
            valid = frame.loc[inside(frame.ra, frame.dec, box, halo)].copy()
            if len(valid):
                valid["in_core"] = inside(valid.ra, valid.dec, box)
                chunks.append(valid)
        if not chunks:
            raise ValueError("No optically selected spectra in the frozen field")
        pd.concat(chunks, ignore_index=True).to_csv(work/f"{stem}.csv", index=False)
    print(f"Frozen rectangle: {len(data):,} detections and {len(gaia):,} Gaia positions", flush=True)
    legacy_cutout(legacy, work/"xiao.csv", box, halo)
    native = json.loads((source/"acquisition_summary.json").read_text())
    save_json(work/"acquisition_summary.json", dict(parent_circular_acquisition=native,
                  selected_detections=len(data), validation_positions=len(gaia),
                  core_area_sqdeg=np.ptp(box["ra"])*np.ptp(np.sin(np.deg2rad(box["dec"])))*180/np.pi))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("work", "source", "archives", "legacy", "config"):
        p.add_argument(name, type=Path)
    args = p.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        p.error("Use a Slurm compute node.")
    name = "south_17_overlap"
    settings = tomllib.loads(args.config.read_text())
    work = args.work/name
    work.mkdir(parents=True, exist_ok=True)
    for local, target in (("parent_manifest.json", args.source/"run_manifest.json"),
                          ("spectra_manifest.json", args.archives/"run_manifest.json")):
        path = work/local
        if not path.exists():
            path.symlink_to(target.resolve(strict=True))
        if path.resolve() != target.resolve(strict=True):
            raise ValueError("Input receipt changed")
    outputs = ("detections.csv", "external_labels.csv", "official_nsc.csv", "validation/gaia.csv",
               "spectra_union.csv", "spectra_veto.csv", "xiao.csv", "acquisition_summary.json")
    with run_stage(work, args.config, name, "overlap-inputs-v11",
                   inputs=("parent_manifest.json", "spectra_manifest.json"), outputs=outputs) as go:
        if go:
            prepare(work, args.source, args.archives, args.legacy, settings["fields"][name], settings["halo_deg"])


if __name__ == "__main__":
    main()
