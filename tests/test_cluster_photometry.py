import numpy as np
import pandas as pd
from deep_pm.cluster_photometry import TRAIN,TRAIN_BACKGROUND,TARGET,BACKGROUND,template


def test_deep_template_ignores_target_photometry_and_motion():
    rng=np.random.default_rng(22)
    train=pd.DataFrame(dict(radius=np.full(180,.27),r0=np.full(180,22.2),color=rng.normal(.46,.035,180)))
    background=pd.DataFrame(dict(radius=np.full(80,.5),r0=np.full(80,22.2),color=rng.uniform(-.2,1.6,80)))
    target=pd.DataFrame(dict(radius=np.full(300,.15),r0=np.full(300,22.2),color=rng.normal(.9,.1,300)))
    phot=pd.concat([train,background,target],ignore_index=True)
    areas=dict(train=1.,training_background=1.)
    original=template(phot,areas)
    phot.loc[phot.radius<.2,'color']=-100
    phot['pmra']=100000.
    assert template(phot,areas)==original
    deep=next(r for r in original if r['lower_r0']==22.)
    assert deep['center_color'] is not None and abs(deep['center_color']-.46)<.11
    rings=sorted([TRAIN,TRAIN_BACKGROUND,TARGET,BACKGROUND])
    assert all(a[1]<b[0] for a,b in zip(rings[:-1],rings[1:]))
