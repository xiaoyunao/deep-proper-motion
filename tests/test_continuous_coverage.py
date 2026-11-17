import json
from pathlib import Path
import tempfile
import unittest

from deep_pm.continuous_coverage import FIT_FILES, audit, planned_cells
from deep_pm.continuous_reference import FILES
from deep_pm.io import digest


class CoverageTest(unittest.TestCase):
    def test_proves_tile_ownership_and_reports_missing_and_wrong_graph_fits(self):
        box = {"ra": [345, 348], "dec": [-1, 0]}
        specs = {k: {"ra": [345 + i, 346 + i], "dec": [-1, 0]}
                 for i, k in enumerate(("one", "two", "three"))}
        with self.assertRaisesRegex(ValueError, "overlapping"):
            planned_cells(box, specs, ("one", "one", "two", "three"))

        def record(path, stage, names, values, state="completed"):
            path.mkdir(parents=True)
            for name in names:
                target = path / name
                target.parent.mkdir(exist_ok=True, parents=True)
                target.write_text(values.get(name, "receipt test data"))
            outputs = {name: digest(path / name) for name in names}
            (path / "run_manifest.json").write_text(json.dumps({
                "stages": {stage: [{"state": state, "outputs": outputs}]}}))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shared = root / "shared"
            shared.mkdir()
            graph = shared / "private_c0.npz"
            graph.write_text("graph")
            plan = [(k, root / "src" / k, root / "ref" / k) for k in specs]
            for k, src, ref in plan[:2]:
                record(src, "exposure-inputs-v09", FILES, {"acquisition_summary.json": json.dumps({
                    "downloaded_nsc_measures": 10, "elapsed_seconds": 5, "peak_rss_mib": 2})})
                record(ref, "continuous-ref-input-v12", ("reference.csv.gz", "reference_summary.json"),
                       {"reference_summary.json": json.dumps({"reference_sources": 3})})
                sha = digest(graph) if k == "one" else "different"
                record(root/"outputs"/k, "continuous-core-v12", FIT_FILES, {
                    FIT_FILES[0]: json.dumps({"reference_gauge": "main", "shared_solution_sha256": sha})})
            (root/"outputs"/"three").mkdir()
            (root/"outputs"/"three"/FIT_FILES[0]).write_text("{}")
            summary, paths = audit(box, specs, plan, root/"outputs", shared)
            self.assertEqual(list(paths), ["one"])
            self.assertEqual(summary["verified_cells"], 1)
            self.assertEqual(summary["acquisition_receipt_cells"], 2)
            self.assertEqual(summary["reference_receipt_cells"], 2)
            self.assertEqual(summary["fit_supported_cells"], 1)
            self.assertEqual(summary["cores"]["one"]["acquisition_resources"]["downloaded_nsc_measures_sum"], 10)
            self.assertEqual(summary["cores"]["one"]["reference_counts"]["reference_sources"], 3)
            self.assertEqual(summary["unverified_cores"], ["two", "three"])
            self.assertEqual(summary["cores"]["two"]["fit"], "wrong_graph")
            self.assertEqual(summary["cores"]["three"]["fit"], "missing_receipt")
            (plan[0][1] / "acquisition_summary.json").write_text("{}")
            changed, paths = audit(box, specs, plan, root/"outputs", shared)
            self.assertEqual(paths, {})
            self.assertEqual(changed["cores"]["one"]["acquisition"], "unverified_summary")
