"""Pool overlap validation on physical observation blocks and apply the frozen gates."""
import argparse
import json
from pathlib import Path

import numpy as np

from .__main__ import tomllib
from .io import json_default
from .validation_blocks import connected

MODELS = ("C0", "C1", "C1b")
CATALOGUES = ("NSC", "Xiao", *MODELS)


def blocked(records, methods, nights, options, seed):
    groups = connected(records, nights)
    result = dict(n=len(records), independent_blocks=len(groups),
                  block_sizes=sorted(map(len, groups), reverse=True), methods=list(methods))
    if not records:
        return result
    vectors = np.asarray([[row["motion"][m] for m in methods] for row in records])
    radial = np.linalg.norm(vectors, axis=2)
    p68 = np.quantile(radial, .68, axis=0)
    result.update(radial_p68_masyr=p68, radial_p95_masyr=np.quantile(radial, .95, axis=0),
                  axes_zero_masyr=np.median(vectors, axis=0))
    b = methods.index("C1b")
    base = [i for i in range(len(methods)) if i != b]
    if np.all(p68[base] > 0):
        result["gain_vs"] = {methods[i]: 1-p68[b]/p68[i] for i in base}
    if len(groups) < options["min_independent_blocks"]:
        return result
    random = np.random.default_rng(seed)
    boot, zero = [], []
    for drawn in random.integers(len(groups), size=(options["resample_draws"], len(groups))):
        i = np.concatenate([groups[j] for j in drawn])
        q = np.quantile(radial[i], .68, axis=0)
        if (q[base] > 0).all():
            boot.append(1-q[b]/q[base])
        zero.append(np.median(vectors[i], axis=0))
    result["axes_zero95"] = np.quantile(zero, [.025, .975], axis=0).transpose(1, 2, 0)
    if boot:
        result["gain95_vs"] = dict(zip((methods[i] for i in base),
                                      np.quantile(boot, [.025, .975], axis=0).T))
    return result


def shared_observations(records):
    pairs = {}
    for site, rows in records.items():
        runs = {e for row in rows for e in row["exposures"] if e.startswith("SDSS:")}
        nsc = {e for row in rows for e in row["exposures"] if not e.startswith("SDSS:")}
        days = {day for row in rows for day in row["nsc_epoch_days"]}
        pairs[site] = (runs, nsc, days)
    result = {}
    for i, (site, values) in enumerate(pairs.items()):
        for other, other_values in list(pairs.items())[i+1:]:
            result[f"{site}__{other}"] = dict(zip(("shared_sdss_runs", "shared_nsc_exposures", "shared_nsc_nights"),
                (len(a & b) for a, b in zip(values, other_values))))
    return result


def site_gate(pilot, options):
    boundary = pilot["overlap_boundary"]
    b = boundary.get("radial_p68_mas", [0, 0, None])
    interval = boundary.get("relative_loss95_vs_c0")
    galaxy = pilot["galaxies"]["comparison"].get("methods", {})
    positions = [galaxy.get(m, {}).get("radial_abs68") for m in MODELS]
    if all(v is not None for v in positions):
        c0, c1, c1b = positions
        gain = c0-c1
        position_pass = (c0-c1b >= gain*options["position_gain_retention"] if gain > 0
                         else c1b <= c0*(1+options["max_position_loss_without_gain"]))
    else:
        position_pass = False
    yields = {m: pilot["source_yield"][m].get("22_to_23", {}).get("final", 0) for m in MODELS}
    return dict(boundary_nominal=bool(b[0] and b[2] <= b[0]*(1+options["max_boundary_loss"])),
                boundary_block_interval=bool(interval is not None and interval[1][1] <= options["max_boundary_upper_loss"]),
                cross_survey_position=bool(position_pass), faint_blind_counts=yields,
                faint_blind_yield=bool(yields["C1"] and yields["C1b"] >=
                                     yields["C1"]*(1-options["max_blind_yield_loss"])))


def compare(data, pilots, ordinary, options, seed):
    result = dict(overlap=shared_observations(data),
                  sites={site: site_gate(pilot, options) for site, pilot in pilots.items()})
    pooled = [row for site in ordinary for row in data[site]]
    for lo, hi in ((21, 23), (21, 22), (22, 23)):
        result[f"{lo}_to_{hi}"] = {}
        for site, rows in dict(ordinary=pooled, **data).items():
            rows = [r for r in rows if lo <= r["magnitude"] < hi]
            full = [row for row in rows if set(CATALOGUES) <= row["motion"].keys()]
            result[f"{lo}_to_{hi}"][site] = {
                model: {group: blocked(subset, methods, nights, options, seed)
                        for group, nights in (("exposure", False), ("night", True))}
                for model, subset, methods in (("three", rows, MODELS), ("five", full, CATALOGUES))}
    faint = result["22_to_23"]["ordinary"]["three"]["night"]
    zero = faint.get("axes_zero95")
    gain = faint.get("gain95_vs", {}).get("C1")
    checks = dict(faint_size=faint["n"] >= options["min_faint_qso"],
                  faint_blocks=faint["independent_blocks"] >= options["min_independent_blocks"],
                  faint_qso_noninferiority=bool(gain is not None and gain[0]>=-options["max_qso_loss"]),
                  faint_qso_c0_gain=faint.get("gain_vs", {}).get("C0", 0)>0,
                  faint_zero_by_axis=bool(zero is not None and
                                         (abs(zero[2]) <= options["zero_limit_masyr"]).all()),
                  ordinary_boundaries=all(result["sites"][x]["boundary_nominal"] and
                                          result["sites"][x]["boundary_block_interval"] for x in ordinary),
                  ordinary_position_gain=all(result["sites"][x]["cross_survey_position"] for x in ordinary),
                  ordinary_yield=all(result["sites"][x]["faint_blind_yield"] for x in ordinary))
    result.update(gate=checks, continuous_production_pass=all(checks.values()))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--ordinary", nargs="+", required=True)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    args = parser.parse_args()
    settings = tomllib.loads(args.config.read_text())
    data, pilots = {}, {}
    for directory in args.runs:
        name = directory.name
        if name in data:
            parser.error("Each field can appear only once")
        data[name] = json.loads((directory/"qso_observation_blocks.json").read_text())
        pilots[name] = json.loads((directory/"pilot_summary.json").read_text())
    if not set(args.ordinary) <= data.keys():
        parser.error("Missing frozen ordinary field")
    output = compare(data, pilots, args.ordinary, settings["overlap"], settings["seed"])
    print(json.dumps(output, default=json_default, allow_nan=False))


if __name__ == "__main__":
    main()
