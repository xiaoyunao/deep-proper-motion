import numpy as np
import pandas as pd

from deep_pm.footprint_qso_readout_v24 import paired


def test_gain_uses_identical_spectral_parent_and_true_vector_norm():
    frame = pd.DataFrame(dict(pm_cov_en=[.1]*8))
    for m, scale in (("ours", .8), ("xiao", 1.), ("nsc", 2.)):
        frame[m+"_pm_e"] = scale*np.arange(1., 9.)
        frame[m+"_pm_n"] = scale*np.arange(2., 10.)
        frame[m+"_se_e"] = frame[m+"_se_n"] = 2.
    for source, variance in (("independent",3.),("reference",1.)):
        for axis in "en":
            frame[source+"_pm_variance_"+axis] = variance
    frame.loc[7,"nsc_pm_e"] = np.nan
    result = paired(frame,("ours","xiao","nsc"))
    assert result["parent"] == 8 and result["common"] == 7
    np.testing.assert_allclose(result["improvement_percent"]["xiao"],20.)
    np.testing.assert_allclose(result["improvement_percent"]["nsc"],60.)
    assert result["catalogues"]["ours"]["tail_gt20"]["n"] == 7
    assert result["our_error_ellipse"]["raw"]["median_squared_mahalanobis"] > 0


def test_annual_parallax_control_uses_only_retained_bright_identical_estimators(tmp_path):
    import duckdb
    from deep_pm.full_coordinate_catalogue import member_key
    from deep_pm.full_coordinate_ngc import final_bright
    path=tmp_path/"catalogue";path.mkdir()
    (path/"v22_catalogue_ownership_summary.json").write_text("{}")
    db=duckdb.connect(str(path/"catalogue.duckdb"))
    db.execute("CREATE TABLE accepted(member_sha256 VARCHAR,magnitude DOUBLE,pm_e DOUBLE,pm_n DOUBLE,se_e DOUBLE,se_n DOUBLE,sdss_runs INT)")
    from deep_pm.full_coordinate_sources import write_rows
    from deep_pm.io import digest, save_json
    db.execute("ALTER TABLE accepted ADD COLUMN core VARCHAR")
    db.execute("ALTER TABLE accepted ADD COLUMN slot INTEGER")
    rows=[];physical=[]
    for i,mag,pm in ((0,18.,1.),(1,19.,1.1),(2,21.,1.),(3,17.,1.)):
        ids=[f"s{i}",f"n{i}"]
        rows.append(dict(original_detection_ids=ids,new={"pm_icrs_masyr":[1.,2.]},
                    new_residual_from_same_estimator_apparent_Gaia_prediction=[.5,-1.],bright_fixed_weights=[.6,1.],
                    bright_fixed_center=[1.,2.],reference_model_sha256='graph'))
        physical.append(dict(detection_ids=ids[::-1],robust_weight=[1.,.6],projection_center_deg=[1.,2.],reference_model_sha256='graph'))
        if i!=3:
            db.execute("INSERT INTO accepted VALUES (?,?,?,?,?,?,?,?,?)",[member_key(ids),mag,pm,2.,1.,1.,1,"c",i])
    db.close()
    folder=tmp_path/"sources/c";folder.mkdir(parents=True)
    file=folder/"private_standard_motion.json.gz"
    write_rows(file,physical)
    save_json(folder/"v22_source_core_summary.json",dict(private_output_sha256={file.name:digest(file)}))
    r=final_bright(tmp_path,rows)
    assert (r["fixed_apparent_predictions"],r["exact_member_matches"],r["usable_observed_r_lt21"],r["changed_estimator"]) == (4,3,2,1)
    assert r["verified_identical_estimator"]["n"] == 1
