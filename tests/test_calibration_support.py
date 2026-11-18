import numpy as np
from deep_pm.precision_v03 import calibration_support


def test_physical_group_names_keep_exact_identity_and_repetitions():
    groups=np.array(['NSC:001','NSC:1','SDSS:42:1','SDSS:42:2','NSC:001','unsupported'],dtype=object)
    offsets={'NSC:001':None,'SDSS:42:2':None}
    np.testing.assert_array_equal(calibration_support(groups,offsets),[True,False,False,True,True,False])
    np.testing.assert_array_equal(calibration_support(groups,{}),np.zeros(len(groups),bool))


def test_hash_query_matches_full_comparisons_for_object_and_unicode_names():
    rng=np.random.default_rng(17)
    names=[f'NSC:{i:05d}' for i in range(300)]
    for dtype in (object,str):
        groups=np.asarray(rng.choice(names,7000),dtype=dtype)
        allowed=names[::3]
        np.testing.assert_array_equal(calibration_support(groups,dict.fromkeys(allowed)),np.isin(groups,allowed))
