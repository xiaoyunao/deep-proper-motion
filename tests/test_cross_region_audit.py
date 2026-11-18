import numpy as np
from deep_pm.astrometry import project
from deep_pm.standard_motion import icrs_motion
from deep_pm.cross_region_audit import frame_state,source_check


def test_independent_projection_changes_do_not_make_physical_motion():
    sky=[188.,29.];motion=np.array([6.,-8.]);rows=[]
    for center in ((183.,33.),(193.,33.)):
        point=project([sky[0]],[sky[1]],center)[0]
        j=icrs_motion(point,np.zeros(2),center)['state_jacobian'][np.ix_([1,3],[1,3])]
        row=icrs_motion(point,np.linalg.solve(j,motion),center)
        rows.append(row)
    np.testing.assert_allclose(frame_state(rows[0]['position_deg'],rows[0]['pm_icrs_masyr'],sky),
                               frame_state(rows[1]['position_deg'],rows[1]['pm_icrs_masyr'],sky),atol=1e-5)


def test_duplicate_position_claims_are_not_common_sources():
    row=dict(position_deg=[188.,29.],pm_icrs_masyr=[1.,2.],detection_ids=['a'],r_bin='22_to_23')
    result=source_check([row,row],[row],True)
    assert result['unique_common_count']==0
    assert 'median' not in result['motion_difference_masyr']
