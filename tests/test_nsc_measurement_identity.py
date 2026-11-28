import json

import pandas as pd
import pytest

from deep_pm.footprint_counted_groups_v24 import GroupCounts
from deep_pm.io import digest
from deep_pm.nsc_measurement_identity import measurement_ids


def test_global_exposure_identity_is_stable_in_a_singleton_halo(monkeypatch, tmp_path):
    path = tmp_path / "rule.json"
    path.write_text(json.dumps(dict(method="qualify-reused-exposure-number-v1",
                                    inventory_sha256="fixture",
                                    exposures={"a": "k4m.17", "b": "k4m.17"})))
    monkeypatch.setenv("DPM_NSC_IDENTITY_RULE", str(path))
    monkeypatch.setenv("DPM_NSC_IDENTITY_SHA256", digest(path))
    frame = pd.DataFrame(dict(measid=["k4m.017.3.94", "k4m.017.3.94", "k4m.531.4.15"],
                              exposure=["a", "b", "c"], ccdnum=[3, 3, 4], objectid=["q", "p", "r"]))
    actual = measurement_ids(frame)
    assert actual.tolist() == ["k4m.017.3.94@a", "k4m.017.3.94@b", "k4m.531.4.15"]
    assert measurement_ids(frame.iloc[1:2]).iloc[0] == actual.iloc[1]
    grouped = GroupCounts(frame.objectid.value_counts())
    grouped.check(frame)
    grouped.finish()
    with pytest.raises(ValueError, match="repeated physical"):
        GroupCounts(pd.Series({"q": 2})).check(frame.iloc[[0, 0]].assign(objectid="q"))
    with pytest.raises(ValueError, match="exposure/detector"):
        measurement_ids(frame.assign(ccdnum=[3, 2, 4]))
    monkeypatch.setenv("DPM_NSC_IDENTITY_SHA256", "invalid")
    with pytest.raises(ValueError, match="unsigned"):
        measurement_ids(frame)


def test_scope_intersects_noncanonical_and_wrapped_longitudes():
    from deep_pm.footprint_identity_scope_v24 import intersects

    box = dict(ra=[0, 1], dec=[4, 5])
    assert intersects(box, [359.95, 359.96, 4.2, 4.3], .06)
    assert intersects(box, [-.05, -.04, 4.2, 4.3], .06)
    assert not intersects(box, [359.8, 359.9, 4.2, 4.3], .06)
    assert not intersects(box, [.3, .5, 5.1, 5.2], .06)


def test_predecessor_comparison_allows_only_the_detection_label_change(monkeypatch, tmp_path):
    from deep_pm.footprint_identity_recovery_v24 import compare_prior

    a, b = tmp_path / "old", tmp_path / "new"
    a.mkdir()
    b.mkdir()
    path = tmp_path / "scope.json"
    path.write_text(json.dumps(dict(method="qualify-reused-exposure-number-v1",
                                    inventory_sha256="fixture", exposures={"e": "k4m.4"})))
    monkeypatch.setenv("DPM_NSC_IDENTITY_RULE", str(path))
    monkeypatch.setenv("DPM_NSC_IDENTITY_SHA256", digest(path))
    old_id, new_id = "NSC:k4m.4.2.8", "NSC:k4m.4.2.8@e"
    for folder, identity in [(a, old_id), (b, new_id)]:
        pd.DataFrame(dict(measid=["k4m.4.2.8"], exposure=["e"], ccdnum=[2], mjd=[56123.125])).to_csv(
            folder / "nsc_object.csv", index=False)
        pd.DataFrame(dict(detection_id=[identity], sigma_east_mas=[75.])).to_csv(folder / "detections.csv", index=False)
        pd.DataFrame(dict(detection_id=[identity], original_object_id=["source"])).to_csv(folder / "external_labels.csv", index=False)
    compare_prior(a, b)
    pd.DataFrame(dict(detection_id=[new_id], sigma_east_mas=[74.])).to_csv(b / "detections.csv", index=False)
    with pytest.raises(AssertionError):
        compare_prior(a, b)
