"""Prepare visit metadata and inspect the published Xiao2025 FITS over HTTP."""
import argparse
from pathlib import Path
import re

import pandas as pd
import requests
from astropy.io import fits

from . import access
from .__main__ import tomllib
from .io import save_json
from .provenance import run_stage

XIAO_URL = "https://china.scidb.cn/download?fileId=189403cb071ff08f299e8f4e082e87fd"


def legacy_header(uri=XIAO_URL):
    # The header fits within four FITS blocks. A 200 could deliver the
    # whole 46.96 GB, so ignore its body and stop if the server ignores Range.
    final = 11519
    with requests.get(uri, headers={"Range": f"bytes=0-{final}", "Accept-Encoding": "identity"},
                      timeout=(12, 30), stream=True) as response:
        content_range = response.headers.get("Content-Range", "")
        expected = re.fullmatch(rf"bytes 0-{final}/([0-9]+)", content_range)
        if response.status_code != 206 or not expected:
            return dict(http_status=response.status_code, range_verified=False)
        data = response.raw.read(final + 1)
    if len(data) != final + 1 or not data.startswith(b"SIMPLE"):
        raise ValueError("The delivered byte range is not a full FITS header sample")
    offset = 0
    while offset < len(data):
        ends = [i for i in range(offset, len(data), 80) if data[i:i+8] == b"END     "]
        if not ends:
            break
        end = ends[0] + 80
        header = fits.Header.fromstring(data[offset:end].decode("ascii"))
        start_data = (end + 2879) // 2880 * 2880
        if header.get("XTENSION") == "BINTABLE":
            cols = {header[f"TTYPE{j}"]: [header[f"TFORM{j}"], header.get(f"TUNIT{j}")]
                    for j in range(1, header["TFIELDS"]+1)}
            return dict(http_status=206, range_verified=True, file_bytes=int(expected[1]),
                        table_data_offset=start_data, row_bytes=header["NAXIS1"],
                        catalogue_size=header["NAXIS2"], columns=cols)
        if header.get("NAXIS", 0) != 0:
            break
        offset = start_data
    raise ValueError("No binary-table header found in the verified initial byte range")


def visit_metadata(work, config, region):
    spec = config["regions"][region]
    data = pd.read_csv(work / "detections.csv", usecols=["survey", "exposure_id"])
    exposure = data.loc[data.survey.eq("NSC"), "exposure_id"].drop_duplicates()
    identifiers = sorted(exposure.str.replace(r"^NSC:", "", regex=True))
    cache = access.Cache(work / "metadata_cache")
    pieces = []
    for start in range(0, len(identifiers), 80):
        rows = identifiers[start:start+80]
        where = "exposure IN (" + ",".join(map(access.literal, rows)) + ")"
        pieces.append(cache.checked("nsc", "exposure,instrument,airmass,mjd",
                                    "nsc_dr2.exposure", where))
    meta = pd.concat(pieces, ignore_index=True)
    if meta.exposure.duplicated().any():
        raise ValueError("NSC exposure metadata has duplicate keys")
    meta.to_csv(work / "nsc_visit_metadata.csv", index=False)
    counts = dict(requested_nsc_visits=len(identifiers), nsc_instruments=meta.instrument.value_counts().to_dict(),
                  missing_airmass=int(meta.airmass.isna().sum()))
    try:
        where = access.bbox(spec["ra"], spec["dec"], 0.12) + " AND class='QSO' AND zWarning=0 AND z>0.5"
        qso = cache.checked("sdss", "specObjID,ra,dec,z", "SpecObj", where, cap=500)
        qso.to_csv(work / "spectroscopic_qso.csv", index=False)
        counts["qso_candidate_count"] = len(qso)
    except (RuntimeError, ValueError) as exc:
        counts["qso_query_error"] = type(exc).__name__
    save_json(work / "visit_metadata_summary.json", counts)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("work", type=Path)
    p.add_argument("kind", choices=["legacy", "visits"])
    p.add_argument("--region", default="test_equator")
    p.add_argument("--config", type=Path, default=Path("configs/precision-v03.toml"))
    args = p.parse_args()
    args.work.mkdir(exist_ok=True, parents=True)
    if args.kind == "legacy":
        out = "legacy_header_summary.json"
        with run_stage(args.work, args.config, args.region, "legacy-header-v03", outputs=[out]) as run:
            if run:
                save_json(args.work / out, legacy_header())
    else:
        outputs = ["nsc_visit_metadata.csv", "spectroscopic_qso.csv", "visit_metadata_summary.json"]
        with run_stage(args.work, args.config, args.region, "visit-metadata-v03",
                       inputs=["detections.csv"], outputs=outputs) as run:
            if run:
                visit_metadata(args.work, tomllib.loads(args.config.read_text()), args.region)


if __name__ == "__main__":
    main()
