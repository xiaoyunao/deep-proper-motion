"""Export verified aggregate JSON from a versioned Scratch run."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import sys

try:
    import tomllib
except ImportError:
    import tomli as tomllib

from .io import save_json


STAGES = {
    'regional-preflight-v07': ('regional-preflight-v07', 'regional_preflight_summary.json',
        ('area_sqdeg', 'joint_fraction', 'nsc_span_fraction', 'faint_qso_positions', 'core_archive_candidates', 'buffered_archive_candidates', 'sampled_positions', 'faint_spectra', 'nscr_exposures_by_instrument', 'single_sdss_fraction', 'repeated_sdss_fraction', 'long_repeat_fraction', 'short_repeat_fraction', 'joint_nsc_span_median')),
    'dense-archive-v07': ('dense-archive-v07', 'dense_archive_summary.json',
        ('area_sqdeg', 'shard_count', 'objects', 'measurements', 'compressed_bytes', 'shards', 'resumed_shards', 'run_elapsed_seconds', 'peak_rss_mib')),
    'regional-production-v07': ('regional-production-v07', 'regional_summary.json',
        ('funnel', 'qso', 'gaia', 'holdout', 'astrometry', 'calibration', 'coverage', 'multiband', 'resources', 'acquisition', 'external_dark_stars')),
    'regional-pool-v07': ('regional-pool-v07', 'regional_cohort_summary.json',
        ('area_sqdeg', 'fields', 'counts', 'aggregate_qso_exits', 'pooled', 'observation_blocks', 'shared_visits', 'quality_candidate_fields', 'gate')),
    'legacy-exit-v07': ('legacy-exit-v07', 'legacy_exit_summary.json',
        ('spectra_in_core', 'photometric_funnel', 'strict_exits', 'association', 'precision')),
    "archive-probe-v06": ("archive-probe-v06", "archive_probe_summary.json",
                          ("area_sqdeg", "per_shard_object_cap", "shards")),
    "astrometry-v06": ("astrometry-v06", "astrometry_summary.json",
                       ("errors", "controls", "coverage", "multiband", "resources")),

    "pilot-pool-v05": ("pilot-pool-v05", "pilot_cohort_summary.json",
                        ("core_area_sqdeg", "fields", "funnel", "final_sources", "final_per_sqdeg",
                         "independent_qso", "v04_stripe_overlap", "pilot_overlap",
                         "resources_by_field", "planning_range")),
    "production-v05": ("production-v05", "production_summary.json",
                        ("core_area_sqdeg", "funnel", "final_per_sqdeg", "calibration",
                         "validation", "holdout", "motion_window", "resources")),
    "nsc-pool-v04": ("nsc-pool-v04", "nsc_cohort_summary.json",
                     ("fields", "pooled", "practical_gain", "observation_blocks", "archive_visit_overlap")),
    "qso-coverage-v04": ("qso-coverage-v04", "qso_coverage_summary.json",
                          ("area_sqdeg", "spectra_core", "spectra_faint_core", "sdss_detection_rows",
                           "sdss_run_count", "sdss_date_count", "nsc_object_rows", "candidates_with_sdss",
                           "candidates_with_both", "faint_candidates_with_both")),
    "nsc-compare-v04": ("nsc-compare-v04", "nsc_comparison_summary.json",
                        ("sample_kind", "counts", "exclusions", "strict", "without_support_requirement",
                         "by_group", "timing_p16_median_p84", "observation_group_influence", "calibration")),
    "legacy-order-v03": ("legacy-order-v03", "legacy_order_summary.json",
                         ("ranges_requested", "ranges_read", "failed_ranges", "sampled_records")),
    "dark-precision-v03": ("dark-precision-v03", "dark_precision_summary.json",
                           ("held_year", "coverage_by_r", "residual_by_r_mas", "qso_coverage",
                            "qso_motion_masyr", "qso_published_common", "paired_object_median_change_mas",
                            "detection_quality_control", "reference_counts")),
    "chromatic-v03": ("chromatic-v03", "chromatic_summary.json",
                      ("status", "exposure_metadata", "models", "comparison", "coverage")),
    "precision-v03": ("precision-v03", "precision_v03_summary.json",
                       ("baseline", "calibration", "common_support", "paired_m3_m2", "exclusion_counts",
                        "temporal", "detection_count", "truth_count", "tail_threshold_masyr")),
    "association": (r"report", "summary/{region}_summary.json",
                    ("inventory", "validation", "resources")),
    "precision": (r"paired-control", "paired_control_summary.json",
                  ("truth_sources", "exclusion_counts", "fixed_truth_three_way",
                   "fixed_truth_common_four_way", "shared_error_sensitivity",
                   "sdss_run_heldout", "sdss_run_camera_residual")),
    "precision-contrasts": (r"paired-control", "paired_contrasts_summary.json", ("paired_nmad",)),
    "matched": (r"matched-control", "matched_control_summary.json",
                ("upstream_commit", "eligible", "fixed_control_total", "matched_pm")),
    "conflict": (r"conflict-review", "conflict_review_summary.json",
                 ("upstream_commit", "eligible", "missed", "missed_with_optimal_candidate",
                  "replacement_delta_quantiles", "conflict_counts")),
    "prediction": (r"prediction-(?:control|held-geometry)", "prediction_control_summary.json",
                   ("held_year", "translated_control")),
    "injection": (r"injection-[0-9]+", "v02_injection_summary.json",
                  ("footprint_visits", "halo_deg", "bins")),
    "stress": (r"stress-[0-9]+", "v02_stress_summary.json",
               ("footprint_visits", "halo_deg", "bins")),
    "challenge-design": (r"prepare-challenge", "challenge_sample_summary.json",
                         ("counts", "bins")),
}
STAGES["pilot-coverage-v05"] = STAGES["qso-coverage-v04"]

for _base in ("regional-preflight","regional-production"):
    _name, _file, _keys = STAGES[_base+"-v07"]
    STAGES[_base+"-v08"] = (_base+"-v08", _file, _keys)
STAGES["spatial-audit-v08"] = ("spatial-audit-v08", "spatial_audit_summary.json",
                              ("qso","morphology","zero","local_gradient","holdout_geometry","m3","resources"))
STAGES["spatial-cohort-v08"] = ("spatial-cohort-v08", "spatial_cohort_summary.json",
                                (*STAGES["regional-pool-v07"][2],"spatial_quality"))
STAGES["spatial-history-v08"] = ("spatial-history-v08", "spatial_history_summary.json",
                                 ("strict_faint_exits","morphology","same_conditions"))
STAGES["exposure-catalogues-v09"] = ("exposure-catalogues-v09","catalogue_audit.json",
    ("legacy_rows","bytes","xiao_columns","xiao_sources","spectra","boxes"))
STAGES["exposure-qso-count-v09"] = ("exposure-qso-count-v09","qso_count_summary.json",
    ("parent_rows","by_radius_arcsec","paper_sample","input_shards"))
STAGES["exposure-preflight-v09"] = ("exposure-preflight-v09","geometry_summary.json",
    ("box","area_sqdeg","tiles","sampled_positions","sampled_ring_pixels","physical_sdss_runs",
     "sdss_scan_centres_buffered","sdss_year_span","geometry_sdss_nights","sdss_visits","nsc_r_visits",
     "nsc_r_exposures","ccd_coverage_by_instrument_band","joint_fraction","joint_timed_fraction",
     "long_sdss_repeat_fraction","objects","approximate_nsc_measurements"))
STAGES["exposure-pilot-v09"] = ("exposure-pilot-v09","pilot_summary.json",
    ("c0","c1","galaxies","qso","gaia","legacy_origin_counts","source_yield","qso_origins",
     "elapsed_seconds","acquisition","gate"))
STAGES["exposure-pilot-v10"] = ("exposure-pilot-v10","pilot_summary.json",
    (*STAGES["exposure-pilot-v09"][2], "gaia_policy", "dark_quality", "reference_graph",
     "physical_boundaries", "sensitivity", "annual_holdout"))
STAGES["validation-existing-v10"] = ("validation-existing-v10", "boundary_summary.json",
    ("sdss_adjacent_field", "held_conditions", "chromatic_trial"))
STAGES["validation-shared-v10"] = ("validation-shared-v10","shared_summary.json",
    ("cores","graph","deduplicated_measurements","supported_groups_used_in_both",
     "shared_sdss_groups","shared_nsc_groups","holdout_references_seen_in_both","calibrated_groups"))

METADATA = {"region", "role", "receipt"}
PRIVATE_KEYS = {"source_id", "source_ids", "object_ids", "objid", "objids", "detection_id",
                "detection_ids", "designation", "track_id", "indices", "ra", "dec",
                "cases", "trace", "rows", "catalogue_rows"}
REGION = re.compile(r"[a-z][a-z0-9_]{0,63}")
HEX = {n: re.compile(rf"[0-9a-f]{{{n}}}") for n in (40, 64)}


def check_aggregate(value):
    if isinstance(value, dict):
        if PRIVATE_KEYS & value.keys():
            raise ValueError("Aggregate contains individual catalogue fields")
        for item in value.values():
            check_aggregate(item)
    elif isinstance(value, list):
        for item in value:
            check_aggregate(item)


def check_payload(kind, region, payload):
    if not REGION.fullmatch(region) or payload.get("region") != region:
        raise ValueError("Unexpected region")
    _, _, keys = STAGES[kind]
    extras = {"sdss_route", "sdss_fields"} if kind in ("qso-coverage-v04", "pilot-coverage-v05") else set()
    if set(payload) - METADATA - extras != set(keys):
        raise ValueError("Unexpected aggregate schema")
    if not REGION.fullmatch(payload["role"]):
        raise ValueError("Unexpected cohort role")
    receipt = payload["receipt"]
    if set(receipt) != {"commit", "config_sha256", "aggregate_sha256", "slurm_job", "stage"}:
        raise ValueError("Unexpected receipt schema")
    for key, length in (("commit", 40), ("config_sha256", 64), ("aggregate_sha256", 64)):
        if not HEX[length].fullmatch(receipt[key]):
            raise ValueError("Malformed revision or checksum")
    if not re.fullmatch(STAGES[kind][0], receipt["stage"]):
        raise ValueError("Receipt belongs to a different stage")
    job = receipt["slurm_job"]
    if job is not None and not re.fullmatch(r"[0-9]+(?:_[0-9]+)?", job):
        raise ValueError("Unexpected Slurm job identifier")
    check_aggregate(payload)


def verify_stage_hashes(stages):
    seen = {}
    for attempts in stages.values():
        if not attempts or attempts[-1]["state"] != "completed":
            continue
        stage = attempts[-1]
        for kind in ("inputs", "outputs"):
            for name, checksum in stage.get(kind, {}).items():
                if seen.setdefault(name, checksum) != checksum:
                    raise ValueError("Completed stages disagree on an input or output checksum")


def export(run_root, kind, region):
    if not REGION.fullmatch(region):
        raise ValueError("Invalid region name")
    if kind == "legacy-order-v03":
        directory = run_root / "legacy"
    else:
        direct = ("challenge-design", "exposure-catalogues-v09", "exposure-qso-count-v09", "validation-shared-v10")
        directory = run_root if kind in direct else run_root / region
    stage_pattern, relative, fields = STAGES[kind]
    relative = relative.format(region=region)
    manifest = json.loads((directory / "run_manifest.json").read_bytes())
    if manifest["region"] != region:
        raise ValueError("Manifest belongs to a different region")
    config_bytes = (directory / "config_snapshot.toml").read_bytes()
    config_hash = sha256(config_bytes).hexdigest()
    if config_hash != manifest["config_sha256"]:
        raise ValueError("Frozen configuration checksum differs")
    completed = [(name, attempts[-1]) for name, attempts in manifest["stages"].items()
                 if re.fullmatch(stage_pattern, name) and attempts
                 and attempts[-1]["state"] == "completed" and relative in attempts[-1]["outputs"]]
    if len(completed) != 1:
        raise ValueError("Exactly one completed stage must record this aggregate")
    verify_stage_hashes(manifest["stages"])
    name, attempt = completed[0]
    summary_bytes = (directory / relative).read_bytes()
    sha = sha256(summary_bytes).hexdigest()
    if sha != attempt["outputs"][relative]:
        raise ValueError("Aggregate checksum differs from completed stage")
    aggregate = json.loads(summary_bytes)
    if kind == "exposure-catalogues-v09":
        aggregate["legacy_rows"] = aggregate.pop("rows")
    if kind == "exposure-preflight-v09":
        for bounds in [aggregate["box"],*(v["region"] for v in aggregate["tiles"])]:
            bounds["ra_bounds_deg"] = bounds.pop("ra")
            bounds["dec_bounds_deg"] = bounds.pop("dec")
    config = tomllib.loads(config_bytes.decode())
    role = "challenge_design" if kind == "challenge-design" else config.get("regions", {}).get(region, {}).get("role", "independent")
    payload = {key: aggregate[key] for key in fields}
    if kind in ("qso-coverage-v04", "pilot-coverage-v05"):
        payload.update(sdss_route=aggregate.get("sdss_route", "cas"), sdss_fields=aggregate.get("sdss_fields"))
    if kind == "dark-precision-v03":
        cohorts = payload["qso_published_common"]
        for table in (cohorts, cohorts.get("ground_catalogue", {})):
            if "metrics" in table:
                table["metrics"] = {key: value if value["n"] >= 3 else {"n": value["n"]}
                                    for key, value in table["metrics"].items()}
    payload.update(region=region, role=role, receipt=dict(
        commit=manifest["commit"], config_sha256=config_hash, aggregate_sha256=sha,
        slurm_job=attempt.get("slurm_job"), stage=name))
    check_payload(kind, region, payload)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["export", "accept"])
    parser.add_argument("kind", choices=[*STAGES, "challenge-results"])
    parser.add_argument("region")
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    if args.kind == "challenge-results":
        if args.region != "challenge":
            raise ValueError("Challenge results must be exported as one cohort")
        if args.command == "export":
            from .challenge_results import export_challenge

            print(json.dumps(export_challenge(args.directory), allow_nan=False))
        else:
            payload = json.load(sys.stdin)
            check_aggregate(payload)
            cohort = payload["cohort"]
            if cohort not in {"gaia_challenge", "gaia_challenge_tail"}:
                raise ValueError("Unexpected challenge cohort")
            if any(row["unknown_truth_status"] for row in payload["bins"].values()):
                raise ValueError("Public challenge results require completed reports for all frozen targets")
            save_json(args.directory / f"v02_{cohort}_results_summary.json", payload)
    elif args.command == "export":
        payload = export(args.directory, args.kind, args.region)
        if args.kind == "association" and payload["validation"]["challenge_target_requested"]:
            raise ValueError("Export only cohort-level results for single-target challenge cones")
        print(json.dumps(payload, allow_nan=False))
    else:
        payload = json.load(sys.stdin)
        check_payload(args.kind, args.region, payload)
        if args.kind == "association" and payload["validation"]["challenge_target_requested"]:
            raise ValueError("Accept only cohort-level results for single-target challenge cones")
        receipt = payload["receipt"]
        stamp = receipt["commit"][:12] + "_" + receipt["config_sha256"][:12]
        suffix = re.search(r"-(v(?:0[3-9]|10))$", args.kind)
        series = suffix.group(1) if suffix else "v02"
        target = args.directory / f"{series}_{args.region}_{args.kind.replace('-', '_')}_{stamp}_summary.json"
        if target.is_file() and json.loads(target.read_bytes()) != payload:
            raise ValueError("A different aggregate already exists for this run")
        save_json(target, payload)


if __name__ == "__main__":
    main()
