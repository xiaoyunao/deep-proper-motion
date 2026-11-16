"""Publish only split- and speed-bin aggregates for single-target challenge cones."""
from collections import Counter, defaultdict
from csv import DictReader
from hashlib import sha256
from io import StringIO
import json
from math import isfinite
import re

from .io import digest
from .public_summary import check_aggregate, export, tomllib
from .validation import ratio


METHODS = ("fixed", "nsc_like", "motion")
STAGES = ("generated", "screened", "retained", "conflict_optimal", "selected_optimal",
          "secure", "pm_compatible_secure")


def summarize_bin(frozen, validations, low, high, coverage_only=0):
    eligible = sum(item["truth_cross_survey"] for item in validations)
    selected = {}
    for method in METHODS:
        selections = [item["methods"][method] for item in validations]
        if any(sum(cell["denominator"] for cell in row["pm_bins"]
                   if (cell["low"], cell["high"]) == (low, high)) != n["truth_cross_survey"]
               for row, n in zip(selections, validations)):
            raise ValueError("An eligible challenge target falls outside its frozen speed bin")
        recovered = sum(row["selected_recall"]["n"] for row in selections)
        selected[method] = dict(among_eligible=ratio(recovered, eligible),
                                among_frozen=ratio(recovered, frozen))
    return dict(frozen=frozen, measured=len(validations), coverage_only=coverage_only,
                eligible=ratio(eligible, frozen),
                selected=selected, motion_stages={
                    step: ratio(sum(item["recovery_stages"][step]["total"]["n"]
                                    for item in validations), eligible) for step in STAGES},
                unknown_truth_status=frozen - len(validations) - coverage_only)


def parallax_class(gaia_bytes, source, limit):
    for row in DictReader(StringIO(gaia_bytes.decode())):
        if row["source_id"] == source:
            parallax = float(row["parallax"] or "nan")
            if isfinite(parallax):
                return "large_abs_parallax" if abs(parallax) >= limit else "small_abs_parallax"
    return "no_measured_parallax"


def empty_survey_returns(directory, stages):
    empty = []
    for survey, stage, filename, id_field in (
        ("SDSS", "fetch-sdss", "sdss.csv", "objid"),
        ("NSC", "fetch-object", "nsc_object.csv", "measid"),
    ):
        attempts = stages.get(stage, [])
        if not attempts or attempts[-1]["state"] != "completed":
            return []
        attempt = attempts[-1]
        path = directory / filename
        with path.open(newline="") as file:
            table = DictReader(file)
            valid_header = id_field in {name.lower() for name in table.fieldnames or ()}
            no_rows = next(table, None) is None
        if valid_header and no_rows:
            if digest(path) != attempt["outputs"][filename]:
                raise ValueError("Empty survey response changed after verified retrieval")
            empty.append(survey)
    return empty


def export_challenge(root):
    design = export(root, "challenge-design", "challenge")
    manifest = json.loads((root / "run_manifest.json").read_bytes())
    frozen = manifest["stages"]["prepare-challenge"][-1]
    config_bytes = (root / "challenge_regions.toml").read_bytes()
    config_hash = sha256(config_bytes).hexdigest()
    if config_hash != frozen["outputs"]["challenge_regions.toml"]:
        raise ValueError("Challenge region configuration changed after sample freeze")
    targets_rel = "validation/challenge_targets.csv"
    target_bytes = (root / targets_rel).read_bytes()
    if sha256(target_bytes).hexdigest() != frozen["outputs"][targets_rel]:
        raise ValueError("Challenge target table changed after sample freeze")
    targets = {row["region"]: row for row in DictReader(StringIO(target_bytes.decode()))}
    config = tomllib.loads(config_bytes.decode())
    prefix = config["challenge"].get("role_prefix", "gaia_challenge")
    limit = config["gaia_max_parallax_for_linear_truth"]
    edges = dict(zip(design["bins"], design["bins"][1:]))
    design_counts = Counter()
    rows = {}
    coverage_counts = Counter()
    coverage_by_survey = Counter()
    distance_groups = defaultdict(list)
    versions = set()
    for name, region in config["regions"].items():
        if region["role"] not in {prefix + "_dev", prefix + "_test"}:
            continue
        match = re.fullmatch(r"hm_(dev|test)_([0-9]+)_[0-9]+", name)
        if match is None:
            raise ValueError("Unexpected frozen challenge region format")
        split, low = match[1], int(match[2])
        if region["role"] != prefix + "_" + split:
            raise ValueError("Challenge region role and split disagree")
        key = split, low, edges[low]
        target = targets[name]
        if (target["split"], int(target["bin_low"]), int(target["bin_high"])) != key:
            raise ValueError("Challenge target and region disagree on the frozen speed bin")
        design_counts[f"{split}/{low}-{edges[low]}"] += 1
        if not (root / name / "run_manifest.json").is_file():
            continue
        run = json.loads((root / name / "run_manifest.json").read_bytes())
        if run["region"] != name:
            raise ValueError("Challenge region and stage receipt disagree")
        report = run["stages"].get("report", [])
        if not report or report[-1]["state"] != "completed":
            empty = empty_survey_returns(root / name, run["stages"])
            if empty:
                versions.add((run["commit"], run["config_sha256"]))
                coverage_counts[key] += 1
                for survey in empty:
                    coverage_by_survey[f"{split}/{survey}"] += 1
            continue
        target_hash = sha256((target["source_id"] + "\n").encode()).hexdigest()
        inputs = run["stages"]["validate"][-1]["inputs"]
        if inputs["validation/challenge_source.txt"] != target_hash:
            raise ValueError("Validated challenge target differs from the frozen sample")
        gaia_rel = "validation/gaia.csv"
        gaia = (root / name / gaia_rel).read_bytes()
        if sha256(gaia).hexdigest() != inputs[gaia_rel]:
            raise ValueError("Gaia validation input changed after the challenge run")
        distance_key = parallax_class(gaia, target["source_id"], limit)
        summary = export(root, "association", name)
        receipt = summary["receipt"]
        versions.add((receipt["commit"], receipt["config_sha256"]))
        validation = summary["validation"]
        if not validation["challenge_target_requested"] or validation["truth_cross_survey"] not in (0, 1):
            raise ValueError("Challenge validation did not use the frozen single-target design")
        rows.setdefault(key, []).append(validation)
        distance_groups[split + "/" + distance_key].append(validation)
    if design_counts != Counter(design["counts"]):
        raise ValueError("Frozen challenge regions disagree with the design counts")
    if len(versions) > 1 or any(h != config_hash for _, h in versions):
        raise ValueError("Inconsistent challenge run configuration or analysis revision")
    report = dict(cohort=prefix, design_receipt=design["receipt"],
                  analysis_commit=next(iter(versions))[0] if versions else None,
                  analysis_config_sha256=config_hash, bins={},
                  parallax_threshold_mas=limit, by_parallax={},
                  coverage_only_zero_survey_returns=dict(coverage_by_survey))
    for split in ("dev", "test"):
        for low, high in edges.items():
            name = f"{split}/{low}-{high}"
            report["bins"][name] = summarize_bin(
                design_counts[name], rows.get((split, low, high), []), low, high,
                coverage_only=coverage_counts[split, low, high])
    for name, validations in sorted(distance_groups.items()):
        n = sum(item["truth_cross_survey"] for item in validations)
        report["by_parallax"][name] = dict(
            measured=len(validations), eligible=ratio(n, len(validations)), selected={
                method: ratio(sum(item["methods"][method]["selected_recall"]["n"]
                                  for item in validations), n) for method in METHODS})
    if not re.fullmatch(r"gaia_challenge(?:_tail)?", prefix):
        raise ValueError("Unexpected challenge cohort")
    check_aggregate(report)
    return report
