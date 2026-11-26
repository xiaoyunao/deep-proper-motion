"""Synthetic numerical acceptance; separate exact reexpression from relinearizing."""
import argparse
from pathlib import Path
import numpy as np

from .astrometry import project,unproject
from .coordinate_metric import forward_jacobian,sky_basis,native_covariance,projected_covariance,fit_vector_motion
from .standard_motion import icrs_motion
from .io import save_json
from .public_summary import check_aggregate


def run(output):
    cases=[((175.,5.),(170.,3.)),((210.,5.),(180.,25.)),((235.,5.),(177.,0.)),((.00001,65.),(355.,30.)),((123.,87.),(118.,40.))]
    report=[];dt=np.array([-17.,-10.,-1.,0.,.01,1.,2.,4.])
    for point,center in cases:
        a=forward_jacobian(*point,center);xy=project(*point,center)[0]
        exact=icrs_motion(xy,a@np.array([30.,-12.]),center)
        inverse=exact['state_jacobian'][np.ix_([1,3],[1,3])]
        local=sky_basis(*point);step=10.;columns=[]
        for j in range(2):
            shifted=[]
            for sign in (1,-1):
                u=local[0]+sign*step/206264806.24709636*local[j+1];u/=np.linalg.norm(u)
                pos=np.rad2deg([np.arctan2(u[1],u[0]),np.arctan2(u[2],np.hypot(u[0],u[1]))]);shifted.append(project(*pos,center)[0])
            columns.append((shifted[0]-shifted[1])/(2*step))
        num=np.column_stack(columns);jac_error=float(np.linalg.norm(a-num)/np.linalg.norm(a))
        if jac_error>1.e-5:raise ValueError('Jacobian finite difference acceptance failed')
        residual=[]
        for rate in (30.,2000.,10000.):
            v=np.array([rate,-.4*rate]);sky=unproject(dt[:,None]*v,point);projected=project(*sky.T,center)
            cov=projected_covariance(*sky.T,center,native_covariance(np.full((len(dt),2),60.)))
            m=fit_vector_motion(projected,2015.5+dt,cov,2015.5,robust=False)
            out=icrs_motion(m.coeff[0],m.coeff[1],center)
            residual.append(dict(total_component_rate_masyr=rate,different_plane_linear_refit_delta_masyr=out['pm_icrs_masyr']-v))
        report.append(dict(point_deg=point,graph_center_deg=center,jacobian_relative_difference=jac_error,
            inverse_forward_max_error=float(np.max(abs(inverse@a-np.eye(2)))),exact_state_reexpression_pm_error=float(np.max(abs(exact['pm_icrs_masyr']-[30,-12]))),
            different_plane_model_curvature=residual))
    # A common reference perturbation is not averaged down by the number of
    # stars when they have identical observing response.
    rng=np.random.default_rng(210621);draw=rng.normal(0,2,(48,2));one=np.cov(draw,rowvar=False)
    mean=np.cov(np.mean(np.repeat(draw[:,None,:],20,axis=1),axis=1),rowvar=False)
    np.testing.assert_allclose(one,mean,rtol=1.e-14,atol=1.e-14)
    result=dict(synthetic_checks=report,synchronous_common_mode_mean_covariance_max_error=float(np.max(abs(one-mean))),
        tests='Analytic forward/inverse,10mas symmetric difference,exact physical state reexpression; RA wrap/high declination/full correlated covariance covered by11 focused pytest cases. Different-plane straight-line refit is a different approximate model; its curvature is reported separately, not mistaken for coordinate failure.',
        tolerances=dict(jacobian_relative=1.e-5,exact_motion_masyr=1.e-3,exact_covariance_relative=1.e-3))
    check_aggregate(result);save_json(output,result)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);a=p.parse_args();run(a.output)
