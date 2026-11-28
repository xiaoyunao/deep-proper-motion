from deep_pm import footprint_dense_v24 as dense
from deep_pm.regional_inputs import GAIA_COLUMNS
import pandas as pd


def test_preflight_populates_canonical_chip_and_gaia_queries(monkeypatch):
    calls=[]
    def checked(cache,route,columns,table,condition,south,north):
        calls.append((route,columns,table,condition,south,north))
        return pd.DataFrame(dict(exposure=['a'],ccdnum=[1]))
    monkeypatch.setattr(dense.access,'checked_spatial',checked)
    spec={'ra':[359,360],'dec':[-2,-1]}
    halo=.04
    assert dense.stage_auxiliary(None,spec,halo)==dict(chip_rows=1,gaia_validation_rows=1)
    for query,pad in zip(calls,[halo+.65,halo]):
        assert query[0]=='nsc'
        assert query[3:]==(dense.where_box(spec,pad),-2-pad-1e-7,-1+pad+1e-7)
    assert calls[0][1:3]==('exposure,ccdnum,instrument,airmass','nsc_dr2.chip')
    assert calls[1][1:3]==(GAIA_COLUMNS+',visibility_periods_used','gaia_dr3.gaia_source')


def test_direct_absence_uses_original_queries_and_splits_at_250():
    from deep_pm.exposure_inputs import confirm_unmeasured
    from unittest.mock import Mock
    import pytest

    cache = Mock()
    cache.checked.return_value = []
    names = sorted("object_" + str(i) for i in range(251))
    confirm_unmeasured(cache, "nsc", names)
    first, last = [call.args for call in cache.checked.call_args_list]
    assert first[:3] == ("nsc", "objectid", "nsc_dr2.meas")
    assert first[3] == "objectid IN (" + ",".join(map(dense.access.literal, names[:250])) + ")"
    assert last[3] == "objectid IN (" + dense.access.literal(names[-1]) + ")"
    cache.checked.return_value = ["unexpected measurement"]
    with pytest.raises(ValueError, match="published measurement rows"):
        confirm_unmeasured(cache, "nsc", names[-1:])


def test_offline_cache_cannot_contact_archive(tmp_path, monkeypatch):
    import pytest
    def forbidden(*args):
        pytest.fail("Offline verification attempted a network query")
    monkeypatch.setattr(dense.access, "request", forbidden)
    with pytest.raises(FileNotFoundError):
        dense.access.Cache(tmp_path, read_only=True).query("nsc", "SELECT COUNT(*)")


def test_old_auxiliary_requires_real_missing_id_verification(tmp_path, monkeypatch):
    import json
    import pytest
    work, core, footprint = tmp_path, "example", "freeze"
    (work / "batch-receipts").mkdir()
    spec = {"ra": [1,2], "dec": [1,2]}
    aux = work / "dense-auxiliary" / (core + ".json")
    pages = work / "dense-pages" / (core + ".json")
    dense.save_json(aux, dict(core=core, footprint_sha256=footprint))
    dense.save_json(pages, dict(core=core, footprint_sha256=footprint, delivered_measurements=1,
                                archive_objects=2, checked_pages=1, expected_from_archive=2))
    frozen_page_sha = dense.digest(pages)
    monkeypatch.setenv("SLURM_JOB_ID", "test")
    monkeypatch.setenv("DPM_REQUIRE_DENSE_AUXILIARY","1")
    monkeypatch.setattr(dense.access, "checked_spatial", lambda *a: pd.DataFrame({"objectid":["a","b"], "ndet":[1,1]}))
    monkeypatch.setattr(dense, "joined_pages", lambda *a, **kw: iter([pd.DataFrame({"objectid":["b"]})]))
    monkeypatch.setattr(dense, "stage_auxiliary", lambda *a: {})
    calls = []
    monkeypatch.setattr(dense, "confirm_unmeasured", lambda c, r, ids: calls.append((r,ids)))
    task = work, core, spec, footprint
    dense.stage(task, prepare_absences_only=True)
    assert not calls and "absence_receipt_sha256" not in json.loads(aux.read_text())
    dense.stage(task)
    dense.stage(task)
    assert calls == [("nsc", ["a"])]
    assert dense.digest(pages) == frozen_page_sha
    assert json.loads(aux.read_text())["unmeasured_archive_objects"] == 1
    absent = work / "dense-absence" / (core + ".json")
    record = json.loads(absent.read_text())
    record["page_receipt_sha256"] = "wrong"
    dense.save_json(absent, record)
    with pytest.raises(ValueError, match="another page receipt"):
        dense.stage(task)
