import numpy as np
import pandas as pd
from astropy.time import Time

from deep_pm.nsc_opportunity_area import estimate,equal_area_grid
from deep_pm.continuous_summary import area


def test_overlapping_ccds_are_not_independent_exposures_and_nonr_adds_time():
    rows=[]
    for name,band,year in [('r1','r',2010),('r1','r',2010),('Y2','Y',2018)]:
        row=dict(exposure=name,filter=band,mjd=Time(f'{year}-06-01',scale='utc').mjd)
        for j,(ra,dec) in enumerate([(-.1,-.1),(.1,-.1),(.1,.1),(-.1,.1)],1):row['vertex_ra'+str(j)]=ra%360;row['vertex_dec'+str(j)]=dec
        rows.append(row)
    window=pd.DataFrame([dict(RA=0.,DEC=0.,RUN=1.,CAMCOL=1.,NODE=0.,INCL=0.,MU_START=359.8,MU_END=.2,NU_START=-.2,NU_END=.2,epoch=2000.)])
    box=dict(ra=[-.05,.05],dec=[-.05,.05]);result=estimate(pd.DataFrame(rows),window,box,dict.fromkeys(['NSC:r1','NSC:Y2','SDSS:1:1']),.025)
    assert result['layers']['r']['geometric_joint_opportunity_area_sqdeg']==0
    np.testing.assert_allclose(result['layers']['all']['geometric_joint_opportunity_area_sqdeg'],area(box))
    assert result['sdss_margin_stability']


def test_equal_solid_angle_cells_preserve_high_declination_area():
    box=dict(ra=[1.,3.],dec=[69.,71.]);sky,unit=equal_area_grid(box,.05)
    np.testing.assert_allclose(unit*len(sky),area(box))
    assert (sky[:,1]>69).all() and (sky[:,1]<71).all()
