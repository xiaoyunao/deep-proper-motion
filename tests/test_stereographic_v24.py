import numpy as np
import pytest
from deep_pm import stereographic as st
from deep_pm.astrometry import basis, project, unproject, MAS_PER_RADIAN
from deep_pm.coordinate_metric import forward_jacobian, angular_distance
from deep_pm.standard_motion import icrs_motion


def test_full_sky_metric_and_state_covariance():
    centre = st.Stereographic((359.,72.))
    assert st.admitted(115) and not st.admitted(165)
    for ra, dec in [(1.,65.),(176.,-8.),(359.99,-10.)]:
        x=project(ra,dec,centre)[0]
        u=unproject(x,centre)[0]
        np.testing.assert_allclose(u,[ra,dec],atol=1e-10)
        A=forward_jacobian(ra,dec,centre)
        v=np.array([18.,-23.]);cov=np.diag([80.,16.,120.,25.])
        output=icrs_motion(x,v,centre,cov)
        jac=output["state_jacobian"]
        np.testing.assert_allclose(jac[np.ix_([1,3],[1,3])]@A,np.eye(2),atol=1e-13)
        native=np.stack([project(ra+1e-5/np.cos(np.radians(dec)),dec,centre)[0]-x,
                         project(ra,dec+1e-5,centre)[0]-x],axis=1)/36
        np.testing.assert_allclose(A,native,rtol=2e-6,atol=2e-6)
        for k in range(2):
            step=np.eye(2)[k]*40.
            plus=icrs_motion(x+step,v,centre)["pm_icrs_masyr"]
            minus=icrs_motion(x-step,v,centre)["pm_icrs_masyr"]
            np.testing.assert_allclose((plus-minus)/80,jac[[1,3],2*k],rtol=2e-6,atol=2e-14)
        assert np.linalg.eigvalsh(output["covariance"]).min()>0
        chord=angular_distance(x,project(ra,dec+1/3.6e6,centre),centre)
        np.testing.assert_allclose(chord,[1.],atol=1e-7)
    with pytest.raises(ValueError,match="antipode"):
        project(179.,-72.,centre)


def test_native_area_cut_in_stereographic_chart():
    from deep_pm.footprint_supported_area_v24 import accepted_direction
    centre=st.Stereographic((1.,70.))
    for ra,dec in [(359.99,0.),(124.,-25.),(3.,71.)]:
        a=forward_jacobian(ra,dec,centre)
        for se in (11.99,12.01):
            cov=a@np.diag([49.,se**2])@a.T
            assert accepted_direction(*np.radians([ra,dec]),basis(*centre),cov,True)==(se<12)


def test_constant_target_projection_offset_transports_across_a_long_scan():
    import pandas as pd
    from deep_pm.footprint_chart_group_transfer_v24 import compare
    center=st.Stereographic((1.,75.))
    ra=np.linspace(330,390,90)%360;dec=np.full(90,12.)
    old=unproject(project(ra,dec,center)-[100,-50],center)
    frame=pd.DataFrame(dict(group='SDSS:a',reference=list(map(str,range(len(ra)))),
                            ra_deg=ra,dec_deg=dec,old_ra=old[:,0],old_dec=old[:,1]))
    _,tested=compare(frame,center)
    assert len(tested)>30 and tested.observed_span.min()>50
    assert tested.error_mas.max()<1e-5
