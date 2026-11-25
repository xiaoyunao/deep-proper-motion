import numpy as np
from deep_pm.usability_quality import condition,learn,metrics,register_models,split_identity


def make_rows(n,window='old',seed=1):
    rng=np.random.default_rng(seed)
    return [dict(identity=['optical',i/10,1.],window=window,magnitude=21.5,sdss_runs=1,
        nsc_span=2.,leverage=.9,exposures=['SDSS:1','NSC:2'],nsc_epoch_days=[1,2],
        values={'blind_all':[(np.array([1.,-2.])+rng.normal(size=2)*3.6).tolist(),[2.,2.]]}) for i in range(n)]


def test_identity_split_and_spatial_transfer_do_not_train_on_test_motion():
    rows=make_rows(1500);new=make_rows(500,'new',2)
    before=register_models(rows,{'new'})
    for r in new:r['values']['blind_all'][0]=[100000.,100000.]
    after=register_models(rows+new,{'new'})
    assert set(before)==set(after)
    key=next(iter(before));np.testing.assert_array_equal(before[key]['median_zero'],after[key]['median_zero'])
    np.testing.assert_array_equal(before[key]['marginal_total_error_multiplier'],after[key]['marginal_total_error_multiplier'])
    assert split_identity(rows[0]['identity'])==split_identity(rows[0]['identity'])


def test_quality_multiplier_describes_total_residual_and_preserves_raw_motion():
    rows=make_rows(3000);model=learn(rows)
    np.testing.assert_allclose(model['median_zero'],[1.,-2.],atol=.2)
    np.testing.assert_allclose(model['marginal_total_error_multiplier'],1.8,atol=.08)
    report=metrics(rows,model['median_zero'],model['marginal_total_error_multiplier'])
    np.testing.assert_allclose(report['training_center_scale_1sigma'],.682689,atol=.03)
    assert not learn(rows[:30])['supported']
    assert condition(rows[0])=='21__one_run__nsc_lt4__leverage_ge08'


def test_native_spectral_identity_is_preserved_before_union_float_columns():
    import pandas as pd
    from deep_pm.exposure_catalogues import merge_qsos
    a=pd.DataFrame(dict(ra=[10.],dec=[0.],origin=['DR16Q'],native_spectral_id=['J004000.00+000000.0'],original_catalogue_row=[7]))
    b=pd.DataFrame(dict(ra=[20.],dec=[0.],origin=['DESI'],target=[9007199254740993],native_spectral_id=['9007199254740993'],original_catalogue_row=[11]))
    union,_=merge_qsos(a,b)
    assert union.native_spectral_id.tolist()==['J004000.00+000000.0','9007199254740993']
    assert union.original_catalogue_row.tolist()==[7,11]
