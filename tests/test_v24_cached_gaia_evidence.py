import json
import pytest
from deep_pm.footprint_gaia_cached_v24 import evidence
from deep_pm.io import save_json,digest


def test_veto_replay_is_bound_to_positions_controls_and_footprint(tmp_path):
    work=tmp_path
    reg=work/"registration"
    c=reg/"v24_gaia_veto_controls_summary.json"
    save_json(c,dict(footprint="fp",controls=[{}]*5,position_membership_passes=True))
    proof=reg/"v24_independent_gaia_veto_summary.json"
    q=dict(footprint="fp",original_sha256="identical",replayed_sha256="identical",
           controls_sha256=digest(c),reference_file_byte_equal=True)
    save_json(proof,q)
    stage=work/"gaia-veto-trial"/"core"
    stage.mkdir(parents=True)
    table=stage/"gaia.csv";table.write_text("source_id,ra,dec\n")
    p=dict(core="core",footprint="fp",controls_sha256=digest(c),gaia_sha256=digest(table))
    save_json(stage/"private_provenance.json",p)
    v=evidence(work,"core","fp")
    assert v["file"]==table and digest(table) in v["config"] and digest(proof) in v["config"]
    q["reference_file_byte_equal"]=False;save_json(proof,q)
    with pytest.raises(ValueError):evidence(work,"core","fp")
    q["reference_file_byte_equal"]=True;save_json(proof,q);table.write_text("changed")
    with pytest.raises(ValueError):evidence(work,"core","fp")
