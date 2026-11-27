import pandas as pd
import pytest

from deep_pm import footprint_sdss_stage_v24 as stage


def test_stage_checks_membership_input_identity_and_csv(tmp_path, monkeypatch):
    sas = tmp_path/"sas"
    key = (123, 301, 4, 5)
    field = stage.sdss_sas.photoobj_path(sas, key)
    field.parent.mkdir(parents=True)
    field.write_bytes(b"field")
    idx = sas/"resolve/2013-07-29/window_flist.fits"
    idx.parent.mkdir(parents=True)
    idx.write_bytes(b"index")
    monkeypatch.setenv("SLURM_JOB_ID", "1")
    monkeypatch.setattr(stage.sdss_sas, "sdss_field_keys", lambda *args: [key])
    rows = pd.DataFrame({"ra": [359.985, .020, .50], "dec": [0., 0., 0.]})
    monkeypatch.setattr(stage.sdss_sas, "read_fields", lambda *args: (rows, 1))
    spec = {"ra": [0., .1], "dec": [-.1, .1]}
    st = tmp_path/"sdss-stage/a"
    dest = tmp_path/"sdss.csv"
    args = (st, dest, sas, spec, .06)
    assert stage.adopt(*args, footprint="test") is None
    assert stage.one((tmp_path, "a", spec, sas, .06, "test", False))[:2] == (2, 1)
    assert stage.adopt(*args, footprint="test") == (2, 1)
    assert pd.read_csv(dest).ra.tolist() == [359.985, .020]
    with pytest.raises(ValueError, match="geometry"):
        stage.adopt(*args, footprint="other")
    with (st/"sdss.csv").open("ab") as f:
        f.write(b"bad\n")
    previous = dest.read_bytes()
    with pytest.raises(ValueError, match="checksum"):
        stage.adopt(*args, footprint="test")
    assert dest.read_bytes() == previous
    field.write_bytes(b"changed field")
    with pytest.raises(ValueError, match="FITS inputs changed"):
        stage.adopt(*args, footprint="test")
