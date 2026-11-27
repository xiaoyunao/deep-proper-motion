import numpy as np
import pandas as pd

from deep_pm.footprint_chart_replay_v24 import prepare, solve, held_readout
from deep_pm.stereographic import Stereographic


def test_identical_physical_groups_and_held_sources_across_chart():
    rows=[]
    for i in range(80):
        for j,g in enumerate(['NSC:a','NSC:b','SDSS:c']):
            rows.append(dict(reference=str(i),detection_id=f'{i}-{j}',exposure_id=g,group=g,
                source_core='a',is_holdout=(i>=60),ra_deg=(359.9+i*.002)%360,
                dec_deg=2+i*.001,sigma_e=55.,sigma_n=60.))
    f=pd.DataFrame(rows)
    train,held,names=prepare([f,f.assign(source_core='b')])
    assert len(train)==180 and len(held)==60
    solutions=[solve(train,names,c,1) for c in [(0.,2.),Stereographic((1.,74.))]]
    residuals,n=held_readout(held,names,solutions)
    assert n==60 and residuals.shape==(20,2,2)
    np.testing.assert_allclose(residuals,0,atol=1e-6)
