"""Residual DCR geometry and SDSS photometric color errors; no new SDSS positions."""
from pathlib import Path

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import AltAz,EarthLocation,SkyCoord
from astropy.io import fits
from astropy.time import Time
from astropy.utils import iers

from .astrometry import basis,MAS_PER_RADIAN
from .io import digest

# Mamajek 2013, arXiv:1210.1616v3 Table 6: WGS84 geodetic, ellipsoid height.
BLANCO=(-70-48/60-23.49/3600,-30-10/60-10.78/3600,2241.4)


def zenith_vectors(mjd_utc):
    iers.conf.auto_download=False
    location=EarthLocation.from_geodetic(BLANCO[0]*u.deg,BLANCO[1]*u.deg,BLANCO[2]*u.m)
    stamp=Time(np.asarray(mjd_utc),format='mjd',scale='utc')
    sky=SkyCoord(az=np.zeros(len(stamp))*u.deg,alt=np.full(len(stamp),90)*u.deg,frame=AltAz(obstime=stamp,location=location,pressure=0*u.hPa)).icrs
    return sky.cartesian.xyz.value.T


def tangent_geometry(ra,dec,zenith,center):
    """tan(z) towards zenith, differential in the common gnomonic plane.

    Only a predictor; never reapply full atmospheric refraction or aberration to
    catalogue positions. Zenith is expressed in ICRS, catalog directions fixed.
    """
    a,d=np.deg2rad([ra,dec]);n=np.column_stack([np.cos(d)*np.cos(a),np.cos(d)*np.sin(a),np.sin(d)])
    z=np.asarray(zenith);cosz=np.sum(n*z,axis=1)
    if (cosz<=.2).any():raise ValueError('DCR geometry below the frozen altitude support limit')
    derivative=(z-n*cosz[:,None])/cosz[:,None]
    frame=basis(*center);local=n@frame.T;dl=derivative@frame.T
    return (dl[:,1:]*local[:,:1]-local[:,1:]*dl[:,:1])/local[:,:1]**2,1/cosz


def sdss_colors(ids,sas):
    """Read only PSF photometry/errors at exact cached physical row identities."""
    parsed=[]
    for name in ids:
        pieces=str(name).split(':')
        if len(pieces)!=5 or pieces[0]!='SDSS':continue
        try:run,camcol,field,number=map(int,pieces[1:])
        except ValueError:continue
        parsed.append((str(name),run,camcol,field,number))
    groups={}
    for name,run,camcol,field,number in parsed:groups.setdefault((run,camcol,field),[]).append((name,number))
    rows=[];receipts={}
    for (run,camcol,field),part in sorted(groups.items()):
        path=Path(sas)/'photoObj'/'301'/str(run)/str(camcol)/f'photoObj-{run:06d}-{camcol}-{field:04d}.fits'
        with fits.open(path,memmap=True) as f:
            table=f[1].data;lookup={int(v):i for i,v in enumerate(table['ID'])}
            for name,number in part:
                if number not in lookup:raise ValueError('Frozen SDSS color identity absent from physical field')
                j=lookup[number];mag=np.asarray(table['PSFMAG'][j],float);err=np.asarray(table['PSFMAGERR'][j],float)
                color=mag[1]-mag[3];error=np.hypot(err[1],err[3])
                valid=np.isfinite(color+error) and -1<color<4 and 0<err[1]<2 and 0<err[3]<2 and (0<mag[[1,3]]).all() and (mag[[1,3]]<35).all()
                rows.append(dict(detection_id=name,color=color if valid else np.nan,color_error=error if valid else np.nan))
        receipts[str(path)]=digest(path)
    return pd.DataFrame(rows,columns=['detection_id','color','color_error']),receipts


def combine_color(frame):
    valid=frame.loc[np.isfinite(frame.color+frame.color_error)&(frame.color_error>0)]
    if not len(valid):return np.nan,np.nan
    weight=1/valid.color_error.to_numpy()**2
    return float(np.average(valid.color,weights=weight)),float(1/np.sqrt(weight.sum()))
