"""Decompose the frozen v2.2 estimator; never change weights or point values."""
import numpy as np
from .coordinate_metric import covariance_of_estimator,native_covariance,projected_covariance,reference_covariance
from .full_coordinate_sources import saved_reference_response
from .precision_v03 import calibration_groups


def components(h,covariance,weight,tags,jacobian):
    """Full two-axis terms, preserving coordinate mixing and C/w convention."""
    h=np.asarray(h,float);c=np.asarray(covariance,float);weight=np.asarray(weight,float)
    pm=np.asarray(jacobian,float)[[1,3]]@h.reshape(4,-1)
    pm=pm.reshape(2,h.shape[1],2)
    per=np.einsum('pni,nij,qnj->npq',pm,c/weight[:,None,None],pm)
    return {str(g):per[np.asarray(tags)==g].sum(axis=0) for g in sorted(set(tags))},per


def decompose(row,data,time,sigma,solution):
    h,ix=saved_reference_response(row,data,time,sigma)
    cp=projected_covariance(data.ra_deg.iloc[ix],data.dec_deg.iloc[ix],row['projection_center_deg'],native_covariance(sigma[ix]))
    split,per=components(h,cp,row['robust_weight'],data.survey.iloc[ix].to_numpy(str),row['state_jacobian'])
    keys=calibration_groups(data)[ix];common,response=reference_covariance(h,np.stack([solution[k][1] for k in keys],axis=1))
    j=np.asarray(row['state_jacobian']);common=(j@common@j.T)[np.ix_([1,3],[1,3])];response=response@j.T
    independent=np.sum(list(split.values()),axis=0);total=independent+common
    target=np.asarray(row['covariance'])[np.ix_([1,3],[1,3])]
    np.testing.assert_allclose(total,target,rtol=5e-7,atol=2e-5)
    np.testing.assert_allclose(independent,np.asarray(row['independent_icrs_covariance'])[np.ix_([1,3],[1,3])],rtol=5e-7,atol=2e-5)
    return dict(survey_terms=split,reference=common,total=total,maximum_covariance_reconstruction_error=float(abs(total-target).max()),
        per_detection_independent_covariance=per,synchronous_reference_state_draws=response,
        interpretation='Frozen weights and H, full2D C/w conditional noise. Survey terms sum with the synchronous reference term; omitted colour/night/spatial systematics are not estimated here.')
