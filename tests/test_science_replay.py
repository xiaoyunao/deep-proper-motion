from copy import deepcopy
import numpy as np
import pytest
from deep_pm.science_replay import compare_rows


def sample():
    return [dict(detection_ids=['SDSS:a','NSC:a'],pm_icrs_masyr=[1.,2.],covariance=[[1.,0.],[0.,1.]],information=dict(nsc_nights=3,effective_information_groups=dict(physical_group=[3.,4.],observing_night=[2.,3.])))]


def test_only_effective_group_machine_summation_roundoff_is_allowed():
    a=sample();b=deepcopy(a);b[0]['information']['effective_information_groups']['physical_group'][0]=np.nextafter(3.,4.)
    result=compare_rows(a,b)
    assert result['all_source_fields_and_covariances_exact']
    assert result['effective_group_float_elements_changed']==1


@pytest.mark.parametrize('change',['motion','covariance','members','nights'])
def test_scientific_values_and_other_metadata_must_remain_exact(change):
    a=sample();b=deepcopy(a)
    if change=='motion':b[0]['pm_icrs_masyr'][0]=np.nextafter(1.,2.)
    elif change=='covariance':b[0]['covariance'][0][1]=1e-15
    elif change=='members':b[0]['detection_ids'][0]='SDSS:wrong'
    else:b[0]['information']['nsc_nights']=4
    with pytest.raises(ValueError):compare_rows(a,b)
