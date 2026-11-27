"""Conformal reference plane for a connected physical domain wider than one safe TAN chart."""
import numpy as np


class Stereographic(tuple):
    """The two ordinary ICRS centre coordinates with an explicit projection type."""


def from_record(record, key="projection_center_deg"):
    center = tuple(record[key])
    kind = str(record.get("projection_kind", "gnomonic"))
    if kind == "gnomonic":
        return center
    if kind == "stereographic":
        return Stereographic(center)
    raise ValueError("Unknown registered astrometric projection: "+kind)


def kind(center):
    return "stereographic" if isinstance(center, Stereographic) else "gnomonic"


def scale(theta_deg):
    return 2/(1+np.cos(np.deg2rad(theta_deg)))


def admitted(theta_deg):
    # Limit conformal magnification to the transverse stretch already admitted
    # at the historical TAN guard. The gnomonic 85 degree guard stays separate.
    return 0 <= theta_deg < 180 and scale(theta_deg) <= 1/np.cos(np.deg2rad(85.))


def plane(local):
    # Also accepts the non-unit barycentric direction used for Gaia predictions.
    radius = np.linalg.norm(local, axis=-1, keepdims=True)
    denom = radius+local[..., :1]
    if np.any(denom <= 1e-12*radius) or np.any(radius == 0):
        raise ValueError("Stereographic antipode or zero direction")
    return 2*local[..., 1:]/denom


def unit(xy):
    h = np.sum(xy*xy,axis=-1,keepdims=True)/4
    return np.concatenate((1-h,xy),axis=-1)/(1+h)


def state(position,motion,frame,mas):
    """Inverse direction and differential PM, including derivatives of the local sky basis."""
    s = position/mas
    h = np.dot(s,s)/4
    denom = 1+h
    n = unit(s)@frame
    b = frame[1:].T
    ds = (b-np.outer(frame[0]+n,s/2))/denom
    q = ds@motion
    horizontal = np.hypot(n[0],n[1])
    if horizontal < 1e-10:
        raise ValueError("ICRS RA is singular at the pole")
    east = np.array([-n[1]/horizontal,n[0]/horizontal,0.])
    north = np.cross(n,east)
    tangent = np.stack((east,north))
    velocity_jacobian = tangent@ds
    derivative = np.zeros((2,2))
    for i in range(2):
        da = (n[0]*ds[1,i]-n[1]*ds[0,i])/horizontal**2
        dd = ds[2,i]/horizontal
        de = -np.array([n[0]/horizontal,n[1]/horizontal,0.])*da
        dn = -n[2]*east*da-n*dd
        # Mixed derivative of n=(C(1-h)+B s)/(1+h), contracted with motion.
        dq = -((frame[0]+n)*motion[i]+s[i]*q+np.dot(s,motion)*ds[:,i])/(2*denom)
        derivative[:,i] = (np.stack((de,dn))@q+tangent@dq)/mas
    return velocity_jacobian,velocity_jacobian@motion,derivative
