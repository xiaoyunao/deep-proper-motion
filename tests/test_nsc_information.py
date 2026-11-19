import numpy as np
import pandas as pd
import pytest
from deep_pm.nsc_information import split_members


def test_new_night_and_centroid_information_preserve_actual_members():
    data=pd.DataFrame(dict(survey=['SDSS','SDSS']+['NSC']*5,band=['r','r','r','r','z','i','g'],
        observation_time=[1,2,3,4,5,3.1,5.1],exposure_id=['S1','S2','R1','R2','Z1','I1','G1'],detection_id=list('abcdefg')))
    split=split_members([0,1,2,3],[0,1,2,3,4,5,6],data,np.array([0,1,2,3,4,2,4]))
    assert split['A_r'].tolist()==[0,1,2,3]
    assert split['B_new_nights'].tolist()==[0,1,2,3,4,6]
    assert split['B1_one_per_new_night'].tolist()==[0,1,2,3,4]
    assert split['C_all'].tolist()==list(range(7))
    with pytest.raises(ValueError,match='SDSS physical detections changed'):split_members([0,2,3],range(7),data,np.arange(7))
    data.loc[1,'band']='g'
    with pytest.raises(ValueError,match='r only'):split_members([0,1,2,3],range(7),data,np.arange(7))


def test_exposure_chip_edge_intersection_does_not_require_center_coverage():
    from deep_pm.nsc_exposure_complete import intersecting
    from deep_pm.astrometry import unproject
    rows=[]
    for center_x in (.04,.20):
        sky=unproject(np.array([[center_x-.02,-.01],[center_x+.02,-.01],[center_x+.02,.01],[center_x-.02,.01]])*3600000,(359.99,20))
        row=dict(exposure=str(center_x),ccdnum=1)
        for k,(ra,dec) in enumerate(sky,1):row['vertex_ra'+str(k)]=ra;row['vertex_dec'+str(k)]=dec
        rows.append(row)
    selected=intersecting(pd.DataFrame(rows),(359.99,20),.025)
    assert selected.exposure.tolist()==['0.04']


def test_season_arm_keeps_fixed_r_and_excludes_new_nights_inside_an_r_season():
    from deep_pm.nsc_information import with_new_seasons

    data = pd.DataFrame(dict(
        survey=['SDSS','NSC','NSC','NSC','NSC','NSC'],
        band=['r','r','g','i','z','Y'],
        observation_time=np.arange(6),
        exposure_id=list('abcdef'), detection_id=list('abcdef')))
    time = np.array([2000,2015,2015.1,2015.02,2017.,2017.08])
    night = np.array([0,1,2,1,3,4])
    arms = split_members([0,1],np.arange(6),data,night)
    result = with_new_seasons(arms,data,time)
    assert list(result['S_new_seasons']) == [0,1,4,5]
    assert list(result['B_new_nights']) == [0,1,2,4,5]
    assert list(result['C_all']) == list(range(6))
