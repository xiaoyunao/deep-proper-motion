import numpy as np
import pytest
from deep_pm.multiband_error_trial import variance_components,physical_pairs


def test_night_variance_identifiability_requires_both_pair_types():
    with pytest.raises(ValueError,match='not identifiable'):
        variance_components([1,2,3],[10,20,40],[False]*3,[0]*3)


def test_variance_components_recover_independent_and_shared_night_scale():
    rng=np.random.default_rng(45);n=12000
    native=rng.uniform(200,7000,n);different=rng.random(n)>.5
    variance=1.2**2*native+2*25**2+2*different*35**2+100
    broad=np.where(rng.random(n)<.05,10,1)
    delta=rng.normal(size=n)*np.sqrt(variance)*broad
    result=variance_components(delta,native,different,np.full(n,100.))
    np.testing.assert_allclose(result,[1.2,25,35],rtol=.12)


def test_pairs_do_not_reuse_observations_or_duplicate_same_exposure():
    time=np.array([2000.,2000.1,2000.2,2001,2002])
    pairs=physical_pairs(np.arange(5),time,np.array([1,1,1,2,3]),np.array(['a','a','b','c','d']))
    flattened=[i for pair in pairs for i in pair]
    assert len(flattened)==len(set(flattened))
    assert pairs==[(0,2),(1,3)]
