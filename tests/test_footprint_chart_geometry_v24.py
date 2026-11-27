import numpy as np
from deep_pm.footprint_chart_geometry_v24 import angular_bounds


def test_chart_radius_certificate_uses_actual_directions():
    near = angular_bounds(np.array([[350.,0.],[10.,0.],[0.,5.]]))
    assert abs(near["lower_bound_deg"]-10.)<1e-4 and near["sample_upper_bound_deg"]<10.01
    wide = angular_bounds(np.array([[0.,0.],[175.,0.],[45.,20.]]))
    assert wide["lower_bound_deg"]>=87.49
    three = angular_bounds(np.array([[0.,0.],[120.,0.],[240.,0.]]))
    assert three["actual_pair_separation_deg"]<121 and three["lower_bound_deg"]>89.99


def test_registered_graph_reads_bound_both_sides(tmp_path,monkeypatch):
    import json
    import duckdb
    import pandas as pd
    from deep_pm.footprint_chart_geometry_v24 import run
    old = tmp_path/"v2.2/graph"
    old.mkdir(parents=True)
    w = tmp_path/"v2.4"
    partial = w/"chart-witness/test"
    partial.mkdir(parents=True)
    (partial/"v24_component_witness_summary.json").write_text(json.dumps({"old_components_sharing_a_provisional_connection":[[0,2]]}))
    (old/"v22_global_graph_summary.json").write_text("{}")
    pd.DataFrame({"group":["a","b"],"component":[0,2]}).to_parquet(old/"private_group_components.parquet")
    with duckdb.connect(str(partial/"sample.duckdb")) as d:
        d.execute("CREATE TABLE eligible AS SELECT i AS reference, g AS \"group\" FROM range(12) t(i) CROSS JOIN (VALUES('a'),('b')) k(g)")
    with duckdb.connect(str(old/"reference.duckdb")) as d:
        d.execute("CREATE TABLE active AS SELECT * FROM (VALUES ('a',1.,0.),('b',176.,0.)) t(\"group\",ra_deg,dec_deg)")
        d.execute("CREATE TABLE group_map AS SELECT * FROM (VALUES ('a',0),('b',2)) t(\"group\",component)")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK","1")
    monkeypatch.setenv("SLURM_JOB_ID","1")
    run(w,partial)
    result=json.loads((partial/"v24_linked_chart_geometry_summary.json").read_text())
    assert result["old_group_connectivity_by_minimum_reference_identities_per_edge"]["10"]==[[0,2]]
    assert result["angular_tests"][0]["lower_bound_deg"]>87.
