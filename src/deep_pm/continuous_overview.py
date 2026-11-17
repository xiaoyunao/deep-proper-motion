"""Export a browser-readable view of the complete continuous science summary."""
import argparse
from pathlib import Path

from .io import digest, read_summary, save_json
from .public_summary import check_aggregate


def overview(data):
    labels = ("strict__all__21_to_22", "strict__all__22_to_23")
    cores = {}
    for name, item in data["cores"].items():
        core = {key: item[key] for key in ("owned_w0", "observations", "formal_error_masyr", "annual",
                                           "leverage", "gaia_policy", "shared_solution_sha256", "reference_gauge",
                                           "peak_rss_mib", "elapsed_seconds")}
        cohorts = item["qso"]["by_cohort"]
        for key in ("c0", "all_three"):
            core["qso_"+key] = {label: cohorts.get(key, {}).get(label, {}) for label in labels}
        core["bright_gaia_c0"] = item["gaia"]["by_cohort"].get("c0", {}).get("strict__all__all", {})
        cores[name] = core
    reference = {name: {key: item["summary"][key] for key in ("counts", "component_count", "by_survey",
                            "unsupported_groups", "shared_calibrated_groups", "held", "max_rss_mib")}
                 for name, item in data["reference"].items()}
    result = dict(total_core_area_sqdeg=data["total_core_area_sqdeg"],
                  catalogue_count=sum(core["owned_w0"] for core in cores.values()),
                  reference=reference,
                  cohorts={key: data["cohorts"][key] for key in ("all", "ordinary", "transition", "repeat_band")
                           if key in data["cohorts"]},
                  covered_cores=data["covered_cores"], cores=cores, w0_sampling_columns=data["w0_sampling_columns"],
                  gauge_cores=data["gauge_cores"], export_scope="selected_existing_aggregate_fields")
    check_aggregate(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = overview(read_summary(args.summary))
    result["source_summary"] = dict(filename=args.summary.name, sha256=digest(args.summary))
    save_json(args.output, result)


if __name__ == "__main__":
    main()
