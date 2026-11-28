"""Preserve NSC DR2 measurements whose published serial numbers were reused."""
from functools import lru_cache
import json
import os
from pathlib import Path

from .io import digest


@lru_cache(maxsize=2)
def load_exposures(path, expected):
    source = Path(path)
    if not expected or digest(source) != expected:
        raise ValueError("Changed or unsigned NSC exposure-number inventory")
    rule = json.loads(source.read_text())
    if rule["method"] != "qualify-reused-exposure-number-v1" or not rule["inventory_sha256"]:
        raise ValueError("Unknown NSC measurement identity rule")
    return rule["exposures"]


def measurement_ids(table):
    """Return archive IDs except in a shared instrument/expnum namespace.

    Qualification depends on the all-sky exposure inventory, so a singleton
    in a neighboring halo gets the same identity as in a collision page.
    The published measid remains in the unchanged raw measurement table.
    """
    ids = table.measid
    rule = os.getenv("DPM_NSC_IDENTITY_RULE")
    if not rule:
        return ids
    exposures = load_exposures(rule, os.getenv("DPM_NSC_IDENTITY_SHA256"))
    if ids.isna().any() or table.exposure.isna().any():
        raise ValueError("Missing NSC measurement or exposure identity")
    affected = table.exposure.isin(exposures)
    if not affected.any():
        return ids
    part = table.loc[affected, ["exposure", "ccdnum", "measid"]]
    fields = part.measid.str.extract(r"^([A-Za-z0-9]+)\.([0-9]+)\.([0-9]+)\.([0-9]+)$")
    if fields.isna().any().any():
        raise ValueError("Ambiguous exposure has an unrecognized NSC measurement serial")
    namespaces = part.exposure.map(exposures)
    # The archive stores expnum numerically, while measid may retain leading zeroes.
    matching = (fields[0] + "." + fields[1].astype("int64").astype(str)).eq(namespaces)
    if not matching.all() or not fields[2].astype("int64").eq(part.ccdnum).all():
        raise ValueError("Measurement serial and published exposure/detector disagree")
    result = ids.copy()
    result.loc[affected] = part.measid + "@" + part.exposure
    return result
