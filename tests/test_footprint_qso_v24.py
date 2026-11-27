import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from deep_pm import footprint_qso_v24 as q
from deep_pm.exposure_catalogues import coordinates


def sample(ra):
    return pd.DataFrame(dict(ra=ra,dec=0.,external=list(range(len(ra))),valid=True,
                             pm_e=0.,pm_n=0.,se_e=1.,se_n=1.))


def test_competitor_across_zero_and_ambiguous_physical_claim(monkeypatch):
    monkeypatch.setattr(q,"_TREE",cKDTree(coordinates([0.,1.],[0.,0.])))
    frame=sample(np.array([-.1,.15,3600.2])/3600)
    # Two competitors for the same spectrum; keep only the other isolated star.
    assert q.unique(q.candidates(frame,1.,True),True).external.tolist()==[2]
    assert q.unique(q.candidates(frame,.5),False).external.tolist()==[2]
    assert q.candidates(frame.iloc[:0],1,True).empty
    identical=pd.concat([frame.iloc[[2]]]*2,ignore_index=True)
    assert q.unique(q.candidates(identical,1.,True),True).external.tolist()==[2]


def test_spectral_owner_denominator_respects_exact_partial_cell_edges():
    cat=sample(np.array([0,.7,.71,.8,359.999]))
    boxes=[dict(spec=dict(ra=[0,.7],dec=[-1,1])),dict(spec=dict(ra=[.7,.71],dec=[-1,1]))]
    assert q.owns_spectrum(cat,boxes).tolist()==[True,True,False,False,False]
