import sqlite3

import pandas as pd
from deep_pm.exposure_preflight import longitude_bounds
from deep_pm.footprint_manifest_v24 import minus_old
from deep_pm.stripe82_comparison_v24 import pair


def test_meridian_count_predicate():
    db=sqlite3.connect(":memory:")
    db.execute("CREATE TABLE test(ra REAL)")
    db.executemany("INSERT INTO test VALUES (?)",[(359.9,),(.1,),(.6,),(100.,),(-.11,),(-.25,),(360.1,),(360.7,)])
    assert db.execute("SELECT count(*) FROM test WHERE "+longitude_bounds(-.2,.6)).fetchone()==(4,)


def test_owner_rectangle_subtraction():
    new=minus_old(2,0,[(2.25,2.75,.2,.9)])
    assert abs(sum((e-w)*(n-s) for w,e,s,n in new)-.65)<1e-12


def test_both_directions_have_to_be_unique():
    hlc=pd.DataFrame(dict(ra=[359.99999,.00001,1.],dec=[0.,0.,0.]))
    other=pd.DataFrame(dict(ra=[0.,1.],dec=[0.,0.],valid=[True,True]))
    i,j=pair(hlc,other)
    assert i.tolist()==[2] and j.tolist()==[1]


def test_hlc_match_exits_partition_entire_parent():
    from deep_pm.stripe82_comparison_v24 import PAIR_STATES
    h = pd.DataFrame(dict(ra=[1.,2.,3.,4.,5.,5.+.13/3600], dec=[0.]*6))
    c = pd.DataFrame(dict(ra=[1.,2.,3.-.1/3600,3.+.1/3600,5.],
                          dec=[0.]*5,valid=[True,False,True,True,True]))
    i,j,status = pair(h,c,diagnostics=True)
    assert i.tolist()==[0] and j.tolist()==[0]
    assert [PAIR_STATES[s] for s in status] == [
        "matched","invalid_catalogue_motion","forward_competition","outside_radius",
        "reverse_competition","not_mutual"]


def test_stripe82_rejects_an_unrelated_catalogue_registration(tmp_path):
    import json
    import pytest
    from deep_pm.stripe82_comparison_v24 import catalogue_closure
    from deep_pm.io import digest
    catalogue=tmp_path/"catalogue.duckdb"
    catalogue.touch()
    plan=tmp_path/"plan.json"
    plan.write_text('{"registered":"first"}')
    binding=tmp_path/"private_closure_binding.json"
    binding.write_text(json.dumps({"plan_sha256":digest(plan)}))
    proof=tmp_path/"v22_catalogue_ownership_summary.json"
    proof.write_text(json.dumps({"private_input_binding_sha256":digest(binding),"retained_common_epoch_solutions":3}))
    assert catalogue_closure(catalogue,plan)==(digest(proof),3)
    plan.write_text('{"registered":"different"}')
    with pytest.raises(ValueError,match="ownership"):
        catalogue_closure(catalogue,plan)


def test_revised_input_counts_shared_query_bytes_without_changing_receipt(tmp_path):
    import json
    from deep_pm.footprint_progress_v24 import account_linked_caches

    shared = tmp_path/"intake"/"b"/"cache"
    shared.mkdir(parents=True)
    (shared/"page.csv").write_bytes(b"header\n")
    revised = tmp_path/"revisions"/"v"/"intake"/"b"
    revised.mkdir(parents=True)
    (revised/"cache").symlink_to(shared, target_is_directory=True)
    resolved = tmp_path/"resolved"
    resolved.mkdir()
    (resolved/"b.json").write_text(json.dumps({"source": str(revised), "source_sha256": "signed"}))
    rows = {"a": {"raw_bytes": 81}, "b": {"raw_bytes": 30, "input_receipt_sha256": "signed"}}
    assert account_linked_caches(tmp_path, rows) == (1, 7)
    assert rows["a"] == {"raw_bytes": 81} and rows["b"]["working_input_bytes"] == 37
    assert rows["b"]["raw_bytes"] == 30
    assert account_linked_caches(tmp_path, rows) == (1, 7)
    assert rows["b"]["working_input_bytes"] == 37
