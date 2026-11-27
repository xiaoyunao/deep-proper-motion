"""The assessor may use spectra only after global source ownership is frozen."""
import json
from pathlib import Path
import runpy

script = Path(__file__).resolve().parents[1] / "scripts/dispatch-v24-production.py"
dispatch = runpy.run_path(str(script))


def test_frozen_catalogue_and_independent_science_dependencies(tmp_path):
    def tasks():
        return dispatch["assessment"](tmp_path, tmp_path/"spectra.csv", tmp_path/"xiao.fits", dispatch["Stage"])

    assert tasks() == ("waiting for catalogue ownership", [])
    root = tmp_path/"joint"
    (root/"catalogue").mkdir(parents=True)
    (root/"catalogue/v22_catalogue_ownership_summary.json").write_text("{}")
    geometry = tmp_path/"geometry"
    geometry.mkdir()
    for bins in (20, 40):
        for kind, sections in (("early", 1), ("late", 2)):
            (geometry/f"private_{kind}_{bins}.json").write_text(json.dumps({"directories": ["shard"]*sections}))
    label, initial = tasks()
    assert label == "assessment"
    labels = {t.label for t in initial}
    assert {"s82", "spectral-pairing", "source-support", "bright-Gaia", "F2"} <= labels
    assert "area-union-40" not in labels and "F3-F4" not in labels and "F5" not in labels
    assert sum(t.args[0]=="footprint_supported_area_v24" for t in initial)==6
    output = root/"assessment"
    for folder, file in (("s82","v24_stripe82_comparison_summary.json"),
                         ("qso","v24_spectral_pairing_summary.json")):
        (output/folder).mkdir(parents=True)
        (output/folder/file).write_text("{}")
    revised = {s.label for s in tasks()[1]}
    assert {"stellar-scans", "stellar-conditions", "F5", "common-errors", "spectral-statistics"} <= revised
    assert not {"s82","spectral-pairing","stellar-sensitivity","F3-F4"} & revised


def test_pending_memory_and_running_billing_use_separate_denominators(tmp_path):
    accounting = dispatch["intake"]["accounts"]
    namespace = accounting.__globals__
    from unittest.mock import patch
    (tmp_path/"private_allocations.txt").write_text("11\n12\n13\n")
    data = "\n".join([
        "11|PENDING|0||billing=1,cpu=1,mem=56G,node=1|120",
        "12|RUNNING|3600|billing=18,cpu=18,mem=31G,node=1|billing=1,cpu=1,mem=31G,node=1|180",
        "13|CANCELLED|0||billing=1,cpu=1,mem=56G,node=1|120",
    ])
    with patch.dict(namespace, {"command": lambda *args: data}):
        usage = accounting(tmp_path)
    assert usage["11"] == ("PENDING", 0., 112*2/256)
    assert usage["12"] == ("RUNNING", 18/256, 18*3/256)
    assert usage["13"] == ("CANCELLED", 0., 0.)
    assert namespace["pending_threads"]("cpu=2,mem=8192M,node=1") == 16


def test_only_explicitly_held_intake_jobs_release_concurrency():
    runnable = dispatch["intake"]["runnable_jobs"]
    active = {"11", "12", "13", "14"}
    assert runnable(active, "11|JobHeldUser\n12|Priority\n13|Dependency") == {"12", "13", "14"}
    assert runnable(active, "") == active
