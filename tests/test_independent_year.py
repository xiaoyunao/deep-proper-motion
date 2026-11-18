import numpy as np
from astropy.time import Time
from scipy.spatial import cKDTree

from deep_pm.independent_year import choose_year,exclusive


def test_calendar_year_selection_uses_distinct_eligible_visits():
    time=Time(['2014-01-02','2015-01-02','2015-01-03','2015-01-04','2016-01-02','2016-01-03','2017-01-02'],scale='utc').tt.jyear
    exp=np.array(['a','b','b','b','c','d','e'])
    years,held=choose_year(time,exp,np.ones(7,bool),np.ones(7,bool))
    assert held==2016
    assert set(years[years!=held])=={2014,2015,2017}


def test_one_held_detection_cannot_validate_two_training_sources():
    good,nearest,n=exclusive(cKDTree([[0.,0.],[2000.,0.]]),np.array([[1.,0.],[2.,0.],[2001.,0.]]),1000,300)
    assert good.tolist()==[False,False,True]
    assert n==2
