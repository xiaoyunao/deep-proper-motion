from pathlib import Path
import numpy as np
import pandas as pd
from deep_pm.__main__ import tomllib
from deep_pm.exposure_pilot import contrast_samples


def test_stationary_control_records_without_claiming_a_blind_catalogue():
    settings=tomllib.loads(Path('configs/spatial-v08.toml').read_text());config=tomllib.loads(Path('configs/precision-v03.toml').read_text())
    time=np.array([2001.,2014.,2018.]);xy=np.zeros((3,2));sigma=np.full((3,2),50.)
    data=pd.DataFrame(dict(detection_id=['early','late1','late2'],exposure_id=['SDSS:1','NSC:a','NSC:b'],survey=['SDSS','NSC','NSC'],band=['r']*3,morphology=[6,.99,.99],observation_time=(time-2000)*365.25+51544.5))
    labels=np.zeros(3,int);indices=np.arange(3);keys=data.exposure_id.to_numpy();nsc=np.array([False,True,True]);offsets={k:(np.zeros(2),np.zeros((48,2)),100) for k in keys}
    original=pd.Series(['','object','object'],index=data.detection_id);blocks=[]
    result=contrast_samples({0:indices},labels,data,xy,time,sigma,np.ones(3,bool),nsc,settings,config,(keys,),(offsets,),[0,0],np.array([21.5]),np.array(['DR16Q']),np.zeros((1,2)),np.zeros((1,2)),(original,pd.DataFrame()),{},[],{'C0':{}},blocks)
    assert len(blocks)==1 and blocks[0]['blind_track_ids']=={'C0':None}
    assert result['totals']['strict']==1
    assert result['totals']['blind_both_models']==0
