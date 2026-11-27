from deep_pm import footprint_dense_v24 as dense
from deep_pm.regional_inputs import GAIA_COLUMNS
import pandas as pd


def test_preflight_populates_canonical_chip_and_gaia_queries(monkeypatch):
    calls=[]
    def checked(cache,route,columns,table,condition,south,north):
        calls.append((route,columns,table,condition,south,north))
        return pd.DataFrame(dict(exposure=['a'],ccdnum=[1]))
    monkeypatch.setattr(dense.access,'checked_spatial',checked)
    spec={'ra':[359,360],'dec':[-2,-1]}
    halo=.04
    assert dense.stage_auxiliary(None,spec,halo)==dict(chip_rows=1,gaia_validation_rows=1)
    for query,pad in zip(calls,[halo+.65,halo]):
        assert query[0]=='nsc'
        assert query[3:]==(dense.where_box(spec,pad),-2-pad-1e-7,-1+pad+1e-7)
    assert calls[0][1:3]==('exposure,ccdnum,instrument,airmass','nsc_dr2.chip')
    assert calls[1][1:3]==(GAIA_COLUMNS+',visibility_periods_used','gaia_dr3.gaia_source')
