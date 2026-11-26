import numpy as np
from deep_pm.full_coordinate_year_readout import source_interval


def test_repeated_test_detections_keep_their_physical_source_in_the_interval():
    values=np.repeat([[0.,0.],[10.,10.]],100,axis=0)
    out=source_interval(values,np.repeat([0,1],100))
    assert out['physical_source_sampling_units']==2 and out['test_measurements']==200
    np.testing.assert_array_equal(out['median_source_resampling95'],[[0,10],[0,10]])
