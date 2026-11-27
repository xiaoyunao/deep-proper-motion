"""Post-freeze Slurm work for the six science figures and their shared samples."""
import json


def assessment(work, spectra, xiao, stage):
    root = work / "joint"
    catalogue = root / "catalogue"
    closure = catalogue / "v22_catalogue_ownership_summary.json"
    if not closure.is_file():
        return "waiting for catalogue ownership", []
    output = root / "assessment"
    native = root / "native-area"
    figures = output / "figures"
    s82 = output / "s82"
    qso = output / "qso"
    support = output / "v24_source_support_summary.json"
    stellar = s82 / "v24_stripe82_comparison_summary.json"
    paired = qso / "v24_spectral_pairing_summary.json"
    readout = qso / "v24_qso_quality_summary.json"
    proposed = []
    base = work.parent

    def add(label, module, args, files, cpu, gib, minutes, requires=()):
        if all(p.is_file() for p in requires) and not all(p.is_file() for p in files):
            proposed.append(stage(label, [module, *map(str, args)], cpu, gib, minutes, 0, 1))

    def images(prefix):
        return [figures / (prefix + extension) for extension in (".pdf", ".png")]

    add("s82", "stripe82_comparison_v24",
        [base, catalogue / "catalogue.duckdb", root / "registration/private_full_coordinate_plan.json",
         s82, "final-frozen-v24"], [stellar], 6, 32, 150)
    add("spectral-pairing", "footprint_qso_v24", [root, spectra, xiao, qso, "--workers", 8],
        [paired], 8, 46, 240)
    add("source-support", "footprint_source_support_v24", [root, support], [support], 8, 20, 70)
    add("bright-Gaia", "full_coordinate_ngc", [root, "--history-root", base],
        [root / "ngc/v22_fixed_ngc_bright_dark_summary.json"], 1, 12, 50)
    add("stellar-conditions", "stripe82_conditions_v24", [s82],
        [s82 / "v24_stripe82_conditional_summary.json"], 1, 8, 35, [stellar])
    scans = s82 / "v24_stripe82_scan_ablation_summary.json"
    add("stellar-scans", "stripe82_scan_ablation_v24", [root, s82, "--workers", 2],
        [scans], 2, 18, 65, [stellar])
    add("stellar-sensitivity", "stripe82_conditions_v24", [s82, "--scan-sensitivity"],
        [s82 / "v24_stripe82_scan_sensitivity_summary.json"], 1, 8, 35, [scans])
    add("spectral-statistics", "footprint_qso_readout_v24", [qso, readout],
        [readout], 1, 12, 85, [paired])
    add("common-errors", "footprint_public_errors_v24", [root, qso, qso, "--workers", 6],
        [qso / "v24_common_qso_population_summary.json"], 6, 36, 240, [paired])

    # Each polygon shard can finish independently. Fine and coarse integrations
    # have distinct geometry registries and share the solved physical offsets.
    for bins in (40, 20):
        parts = []
        for kind in ("early", "late"):
            registry = work / "geometry" / f"private_{kind}_{bins}.json"
            directories = json.loads(registry.read_text())["directories"]
            if not directories:
                raise ValueError("No native geometry partitions")
            for section in range(len(directories)):
                part = native / f"{kind}-{bins}-{section:03d}.json"
                add(f"area-{kind}-{bins}-{section}", "footprint_supported_area_v24",
                    [base, kind, "--section", section, "--bins", bins], [part], 1, 16, 120)
                parts.append(part)
        area = native / f"v24_native_reference_area_{bins}_summary.json"
        add(f"area-union-{bins}", "footprint_supported_area_v24",
            [base, "union", "--bins", bins], [area], 6, 20, 90, parts)
        area_output = output / f"v24_output_area_{bins}_summary.json"
        add(f"output-area-{bins}", "footprint_output_area_v24",
            [base, area_output, "--bins", bins], [area_output], 4, 16, 90, [area])

    native_area = native / "v24_native_reference_area_40_summary.json"
    add("F1", "footprint_coverage_figure_v24",
        [work / "registration/v24_geometric_sky_summary.json", figures / "F1_coverage",
         "--source-summary", support, "--reference-area", native_area],
        images("F1_coverage"), 1, 5, 15, [support, native_area])
    add("F2", "footprint_trajectory_v24", [root, figures],
        images("F2_trajectories") + [figures / "private_f2_examples.json"], 1, 10, 45)
    add("F3-F4", "footprint_qso_figures_v24", [readout, figures],
        images("F3_qso_precision") + images("F4_qso_spatial_quality"), 1, 5, 20, [readout])
    add("F5", "stripe82_figure_v24", [stellar, figures / "F5_stripe82"],
        images("F5_stripe82"), 1, 4, 15, [stellar])
    add("F6", "footprint_information_figure_v24", [support, figures / "F6_time_support"],
        images("F6_time_support"), 1, 4, 15, [support])
    return ("assessment", proposed) if proposed else ("assessment finished", [])
