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
