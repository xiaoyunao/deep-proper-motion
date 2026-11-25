import numpy as np
import pandas as pd
from deep_pm.usability_candidate import independent_acceptor


def test_independent_source_evidence_collapses_runs_but_rejects_other_source_and_unknown():
    data=pd.DataFrame(dict(survey=['SDSS','SDSS','NSC','SDSS'],
        exposure_id=['a','b','c','a'],morphology=[6,6,.7,6]))
    xy=np.array([[0.,0.],[20.,0.],[0.,0.],[700.,0.]])
    accept=independent_acceptor(data,xy,np.array([1,1,0,0],bool))
    assert accept([2],xy[2])  # Two real runs of one source are independent STAR evidence.
    crowded=independent_acceptor(data,xy,np.array([1,1,0,1],bool))
    assert not crowded([2],np.array([300.,0.]))
    data.loc[2,'morphology']=np.nan
    assert not accept([2],xy[2])
    data.loc[2,'morphology']=.59
    assert not accept([2],xy[2])
    data.loc[2,'morphology']=.7;data.loc[1,'morphology']=3
    assert not independent_acceptor(data,xy,np.array([1,1,0,0],bool))([2],xy[2])
