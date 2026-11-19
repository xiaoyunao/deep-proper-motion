import numpy as np
import pandas as pd
from deep_pm.allband_blind_year import r_magnitude
from deep_pm.independent_year import choose_year
from deep_pm.astrometry import years
from astropy.time import Time


def test_non_r_anchor_uses_actual_sdss_r_magnitude_label():
    data=pd.DataFrame(dict(band=['g','z','r'],magnitude=[20.,23.,22.1]))
    assert r_magnitude(np.arange(3),data,np.array([True,True,False]))==(22.1,'SDSS_r')


def test_all_band_metadata_year_can_hold_out_year_without_any_r():
    stamps=Time(['2014-01-02','2015-06-02','2018-01-02'],scale='utc').tt.mjd
    year,held=choose_year(years(stamps),np.array(['g_early','z_middle','i_late']),np.ones(3,bool),np.ones(3,bool))
    assert held==2015
    np.testing.assert_array_equal(year,[2014,2015,2018])
