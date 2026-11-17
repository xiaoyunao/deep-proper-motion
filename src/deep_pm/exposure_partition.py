"""Plan conservative HEALPix storage pixels with exact rectangular ownership."""
import argparse
import json
from pathlib import Path

import healpy as hp
import numpy as np

from .__main__ import tomllib
from .exposure_catalogues import inside


def storage_pixels(box, nside, halo):
    # Inflate the corner quadrilateral so its great-circle sides contain
    # the requested constant-declination edges of these equatorial boxes.
    lo,hi=box["ra"]
    south,north=box["dec"]
    angle=np.deg2rad((hi-lo+2*halo)/2)
    margin=np.rad2deg(np.arctan(np.tan(np.deg2rad(max(abs(south),abs(north))+halo))
                      / np.cos(angle))) - (max(abs(south),abs(north))+halo)
    pad=halo+margin+.005
    vertex=hp.ang2vec([lo-pad,hi+pad,hi+pad,lo-pad],
                     [south-pad,south-pad,north+pad,north+pad],lonlat=True)
    return hp.query_polygon(nside,vertex,inclusive=True,fact=16,nest=False).tolist()


def owner(ra,dec,box,nside):
    ra,dec=np.asarray(ra),np.asarray(dec)
    valid=inside(ra,dec,box)&np.isfinite(ra)&np.isfinite(dec)
    ids=np.full(ra.shape,-1,dtype=int)
    ids[valid]=hp.ang2pix(nside,ra[valid],dec[valid],lonlat=True,nest=False)
    return ids


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("config",type=Path)
    a=p.parse_args()
    settings=tomllib.loads(a.config.read_text())
    name=settings["selection"]["box"]
    box=settings["boxes"][name]
    out=dict(region=name,ordering="RING",nside=settings["nside"],
             halo_deg=settings["halo_deg"],pixels=storage_pixels(box,settings["nside"],settings["halo_deg"]))
    print(json.dumps(out))


if __name__=="__main__":
    main()
