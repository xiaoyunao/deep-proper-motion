"""Full C0 covariance and differential ICRS output; frozen fits stay unchanged."""
import numpy as np

from .astrometry import MAS_PER_RADIAN, basis, unproject


def calibrated_covariance(model, time, sigma, offsets):
    """State order (x0, mux, y0, muy), mas and mas/yr at model.epoch.

    Conditional IRLS covariance plus synchronous reference-bootstrap covariance.
    The same draw index must describe the same reference solution in every group.
    This linear propagation holds the fitted robust weights fixed, as in v1.2.
    """
    time, sigma = np.asarray(time), np.asarray(sigma)
    design = np.column_stack([np.ones(len(time)), time-model.epoch])
    influence = np.zeros((4, len(time), 2))
    independent = np.zeros((4, 4))
    for axis in range(2):
        sl = slice(2*axis, 2*axis+2)
        independent[sl, sl] = model.cov[axis]
        influence[sl, :, axis] = model.cov[axis] @ (design.T*(model.weight/sigma[:,axis]**2))
    draws = np.stack([row[1] for row in offsets], axis=1)
    if len(draws) < 2 or draws.shape[1:] != (len(time), 2):
        raise ValueError("Need synchronous two-axis reference bootstrap draws")
    propagated = np.einsum('pna,dna->dp', influence, draws)
    common = np.cov(propagated, rowvar=False)
    return independent+common, independent, common, influence


def icrs_motion(position, motion, center, covariance=None):
    """Inverse-gnomonic velocity and full first-order state covariance.

    Input state order: x0, mux, y0, muy. Output covariance order:
    local alpha-star position, pmra, local delta position, pmdec.
    Position uncertainties are local angular displacements in mas, rather than
    covariance of the coordinate ra*cos(dec). PM units are mas per Julian year.
    """
    position, motion = np.asarray(position, float), np.asarray(motion, float)
    if position.shape != (2,) or motion.shape != (2,) or not np.isfinite([position, motion]).all():
        raise ValueError("Need finite two-axis position and velocity")
    frame = basis(*center)
    b = frame[1:].T
    raw = frame[0]+b@position/MAS_PER_RADIAN
    length = np.linalg.norm(raw)
    n = raw/length
    h = np.hypot(n[0], n[1])
    if h < 1e-10:
        raise ValueError("ICRS RA is singular at the pole")
    east = np.array([-n[1]/h, n[0]/h, 0.])
    north = np.array([-n[2]*n[0]/h, -n[2]*n[1]/h, h])
    local = np.stack([east, north])
    projector = np.eye(3)-np.outer(n,n)
    velocity_jacobian = local@projector@b/length
    velocity = velocity_jacobian@motion
    q = b@motion
    s = n@q
    projected = q-n*s
    derivative = np.zeros((2,2))
    for i in range(2):
        dn = projector@b[:,i]/(MAS_PER_RADIAN*length)
        da = (n[0]*dn[1]-n[1]*dn[0])/h**2
        dd = dn[2]/h
        de = np.array([-n[0]/h,-n[1]/h,0.])*da
        d_north = -n[2]*east*da-n*dd
        dlocal = np.stack([de,d_north])
        dlength = position[i]/(MAS_PER_RADIAN**2*length)
        derivative[:,i] = (dlocal@projected-local@(dn*s+n*(dn@q)))/length-velocity*dlength/length
    jacobian = np.zeros((4,4))
    jacobian[np.ix_([0,2],[0,2])] = velocity_jacobian
    jacobian[np.ix_([1,3],[1,3])] = velocity_jacobian
    jacobian[np.ix_([1,3],[0,2])] = derivative
    result = dict(position_deg=unproject(position,center)[0], pm_icrs_masyr=velocity,
                  state_jacobian=jacobian,
                  covariance_order=['alpha_star_position_mas','pmra_masyr','delta_position_mas','pmdec_masyr'])
    if covariance is not None:
        covariance = np.asarray(covariance, float)
        if covariance.shape != (4,4) or not np.isfinite(covariance).all():
            raise ValueError("Need a finite 4 by 4 state covariance")
        result['covariance'] = jacobian@covariance@jacobian.T
        result['pm_error_masyr'] = np.sqrt(np.diag(result['covariance'])[[1,3]])
    return result
