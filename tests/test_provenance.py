from pathlib import Path
import json
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from deep_pm.provenance import run_stage


class ProvenanceTest(unittest.TestCase):
    @patch("deep_pm.provenance.subprocess.check_output", return_value="snapshot")
    def test_distinct_stages_preserve_original_artifacts(self, _):
        with TemporaryDirectory() as folder:
            path = Path(folder)
            config = path / "config.toml"
            config.write_text("seed=1")
            (path / "detections.csv").write_text("detected")
            with run_stage(path, config, "smoke", "associate", ["detections.csv"], ["tracks.json"]):
                (path / "tracks.json").write_text("tracks")
            with run_stage(path, config, "smoke", "report", ["tracks.json"]):
                pass
            manifest = json.loads((path / "run_manifest.json").read_text())
            self.assertIn("detections.csv", manifest["stages"]["associate"][0]["inputs"])
            (path / "tracks.json").write_text("altered")
            with self.assertRaises(ValueError):
                with run_stage(path, config, "smoke", "report", ["tracks.json"]):
                    pass
