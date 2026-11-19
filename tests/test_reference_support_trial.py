import numpy as np
from deep_pm.reference_support_trial import prepare
from deep_pm.astrometry import fit_motion
from deep_pm.standard_motion import calibrated_covariance


def test_support_trial_preserves_all_shared_draws_and_old_values(tmp_path,monkeypatch):
    original=tmp_path/'original';work=tmp_path/'trial';original.mkdir();work.mkdir();monkeypatch.setenv('SLURM_JOB_ID','synthetic')
    b=dict(names=np.array(['NSC:a','NSC:b','SDSS:c']),position=np.arange(6).reshape(3,2),draws=np.arange(48*3*2).reshape(3,48,2),counts=np.array([100,100,100]),gauge=np.array(['0','0','0']),projection_center_deg=np.array([0.,0.]))
    np.savez_compressed(original/'private_c0_unfiltered.npz',**b)
    a={k:v[:2] if k!='projection_center_deg' else v for k,v in b.items()};np.savez_compressed(original/'private_c0.npz',**a)
    prepare(work,original)
    with np.load(work/'private_c0.npz') as result:
        for k,v in b.items():np.testing.assert_array_equal(result[k],v)


def test_common_reference_translation_stays_in_position_not_motion():
    t=np.array([2001.,2014.,2015.,2018.]);sigma=np.full((4,2),50.);xy=np.column_stack([2*(t-2015.5),3*(t-2015.5)])
    model=fit_motion(xy,t,sigma,2015.5)
    rng=np.random.default_rng(4);draws=rng.normal(size=(48,2))*[36,18]
    total,independent,reference,_=calibrated_covariance(model,t,sigma,[(np.zeros(2),draws,100)]*4)
    np.testing.assert_allclose(reference[np.ix_([0,2],[0,2])],np.cov(draws,rowvar=False),rtol=1e-12)
    np.testing.assert_allclose(np.diag(reference)[[1,3]],0,atol=1e-20)
    np.testing.assert_allclose(total,independent+reference)
