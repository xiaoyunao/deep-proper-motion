from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from deep_pm.public_summary import export


class PublicSummaryTest(unittest.TestCase):
    def fixture(self, root, **options):
        region = root / "pal5"
        region.mkdir()
        summary = dict(truth_sources=3, exclusion_counts={}, fixed_truth_three_way={},
                       fixed_truth_common_four_way={}, shared_error_sensitivity={},
                       sdss_run_heldout={}, sdss_run_camera_residual={})
        summary.update(options)
        raw = json.dumps(summary).encode()
        config = b'[regions.pal5]\nrole = "regression"\n'
        (region / "config_snapshot.toml").write_bytes(config)
        (region / "paired_control_summary.json").write_bytes(raw)
        (region / "run_manifest.json").write_text(json.dumps(dict(
            region="pal5", commit="a" * 40, config_sha256=sha256(config).hexdigest(),
            stages={"paired-control": [dict(state="completed", slurm_job="123",
                      outputs={"paired_control_summary.json": sha256(raw).hexdigest()})]})))
        return region

    def test_verified_summary_gets_role_and_receipt_without_private_manifest(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root, internal_source_id="excluded by public schema")
            result = export(root, "precision", "pal5")
            self.assertEqual(result["role"], "regression")
            self.assertEqual(result["receipt"]["slurm_job"], "123")
            self.assertNotIn("internal_source_id", result)
            self.assertNotIn("manifest", result)

    def test_nested_individual_columns_and_tampering_are_rejected(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root, fixed_truth_three_way={"source_id": "42"})
            with self.assertRaisesRegex(ValueError, "individual catalogue"):
                export(root, "precision", "pal5")
        with TemporaryDirectory() as folder:
            root = Path(folder)
            region = self.fixture(root)
            (region / "paired_control_summary.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "checksum differs"):
                export(root, "precision", "pal5")

    def test_pending_job_cannot_export_partial_summary(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            region = self.fixture(root)
            manifest = region / "run_manifest.json"
            source = json.loads(manifest.read_text())
            source["stages"]["paired-control"][-1]["state"] = "running"
            manifest.write_text(json.dumps(source))
            with self.assertRaisesRegex(ValueError, "completed stage"):
                export(root, "precision", "pal5")

    def test_different_populations_in_same_run_are_not_exported(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            region = self.fixture(root)
            manifest = region / "run_manifest.json"
            source = json.loads(manifest.read_text())
            source["stages"]["inventory"] = [dict(state="completed", inputs={"detections.csv": "a"})]
            source["stages"]["paired-control"][-1]["inputs"] = {"detections.csv": "b"}
            manifest.write_text(json.dumps(source))
            with self.assertRaisesRegex(ValueError, "stages disagree"):
                export(root, "precision", "pal5")
