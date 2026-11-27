import pandas as pd
import pytest
import requests

from deep_pm import access
from deep_pm.exposure_inputs import tap_join


def test_join_read_timeout_uses_identical_object_predicate(monkeypatch):
    monkeypatch.setenv("DPM_NSC_TRANSPORT", "counted-tap")
    calls = []

    def timeout(*args, **kwargs):
        calls.append(kwargs["params"]["QUERY"])
        raise requests.exceptions.ReadTimeout("delayed response")

    monkeypatch.setattr(requests, "get", timeout)
    with pytest.raises(access.SlowTapJoin):
        access.request("tap", "SELECT COUNT(*) FROM nsc_dr2.meas m JOIN nsc_dr2.object o ON m.objectid=o.id WHERE x=1")
    assert len(calls) == 1

    class Cache:
        def checked_native_window(self, columns, table, where, cap):
            self.where = where
            raise access.SlowTapJoin()

        def checked(self, service, columns, table, where, cap):
            if table == "nsc_dr2.object o":
                assert where == self.where and "o.ndetr" not in where
                return pd.DataFrame({"objectid": ["b", "a"]})
            assert table == "nsc_dr2.meas" and where == "objectid IN ('a','b')"
            return pd.DataFrame({"measid": ["m1", "m2"], "objectid": ["b", "b"]})

    sky = pd.DataFrame({"ra": [12.], "dec": [4.], "ndet": [10]})
    result = tap_join(Cache(), sky, {"ra": [11., 13.], "dec": [3., 5.]}, .01,
                      service="nsc", workers=1, native_window=True, all_objects=True)
    assert result.measid.tolist() == ["m1", "m2"]
