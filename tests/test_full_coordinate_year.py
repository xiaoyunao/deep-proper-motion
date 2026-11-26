import numpy as np
from deep_pm.full_coordinate_year import paired_entities,prediction_covariance


def test_training_continuity_is_unique_without_using_test_positions():
    old=[dict(members=['a','b','c']),dict(members=['q','r'])]
    new=[dict(members=['a','b','d']),dict(members=['s','t'])]
    out=paired_entities(old,new)
    assert out==[dict(old=0,new=0,kind='common_training_identity'),dict(old=1,new=None,kind='old_only_training'),dict(old=None,new=1,kind='new_only_training')]
    competing=[dict(members=['a','b','e']),dict(members=['a','b','f'])]
    assert all(e['kind']!='common_training_identity' for e in paired_entities(old,competing))


def test_shared_reference_prediction_cancels_synchronously():
    rng=np.random.default_rng(22);z=rng.normal(size=(48,2))*20
    response=np.zeros((48,4));response[:,[0,2]]=-z
    row=dict(epoch=2015.5,independent=np.diag([4,1,9,2]),response=response)
    actual=prediction_covariance(row,2015.5,np.eye(2)*3,z)
    np.testing.assert_allclose(actual,np.diag([7,12]))
    wrong=prediction_covariance(row,2015.5,np.eye(2)*3,z[::-1])
    assert np.trace(wrong)>100
