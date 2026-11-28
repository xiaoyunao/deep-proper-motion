import json

import pandas as pd
import pytest

from deep_pm.footprint_counted_groups_v24 import GroupCounts, independent_counts, query_key, strips


def test_every_measured_object_occurs_once_with_its_independent_frequency():
    expected = pd.Series({"a": 2, "b": 1}, dtype="int64")
    check = GroupCounts(expected)
    data = pd.DataFrame({"objectid": ["a", "a"], "measid": ["1", "2"]})
    with pytest.raises(ValueError, match="frequencies"):
        check.check(data.iloc[:1])
    with pytest.raises(ValueError, match="repeated physical"):
        check.check(pd.concat([data.iloc[:1], data.iloc[:1]]))
    check.check(data)
    with pytest.raises(ValueError, match="repeated object"):
        check.check(data)
    with pytest.raises(ValueError, match="Missing objects"):
        check.finish()
    check.check(pd.DataFrame({"objectid": ["b"], "measid": ["3"]}))
    check.finish()
    assert check.measurements == 3


def test_group_response_is_checked_against_independent_total_including_empty_objects(tmp_path):
    class Cache:
        path = tmp_path
        total = 2

        def query(self, service, sql, expected=None):
            (self.path / (query_key(sql) + ".json")).write_text(json.dumps({"sha256": "example"}))
            if expected == 1:
                return pd.DataFrame({"n": [self.total]})
            return pd.DataFrame({"objectid": ["b"], "n": [2]})

    cache = Cache()
    archive = pd.DataFrame({"objectid": ["a", "b"], "dec": [0.1, 0.2], "ndet": [1, 2]})
    check, binding = independent_counts(cache, archive, "objects", "1=1", 0, 1)
    assert check.counts.to_dict() == {"b": 2} and len(binding) == 2
    cache.total = 3
    with pytest.raises(ValueError, match="Truncated"):
        independent_counts(cache, archive, "objects", "1=1", 0, 1)


def test_independent_count_strips_share_only_the_boundary():
    archive = pd.DataFrame({"dec": [0.1, 0.5], "ndet": [700000, 700000]})
    assert list(strips(archive, 0, 1)) == [(0, 0.5), (0.5, 1)]
