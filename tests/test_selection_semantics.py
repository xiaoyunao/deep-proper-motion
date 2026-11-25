import json
import numpy as np
import pandas as pd

from deep_pm.selection_semantics import morphology_state


def test_missing_morphology_is_not_silently_called_pointlike():
    mixed=morphology_state([.2,np.nan])
    assert mixed['state']=='mixed_valid_missing'
    assert mixed['legacy_pass'] and not mixed['valid_only_pass']
    assert mixed['missing_count']==1 and mixed['valid_count']==1
    assert morphology_state([np.nan,np.nan])['state']=='all_missing'
    assert morphology_state([2.,.95])['invalid_count']==1
    assert morphology_state([.95,.98])['state']=='valid_pointlike'
    assert morphology_state([.2,.3])['state']=='valid_below_threshold'


def test_selection_trace_preserves_nan_legacy_catalogue():
    from deep_pm.allband_qso import evaluated_fit
    from deep_pm.io import json_default
    from deep_pm.nsc_science_catalogue import select_catalogue
    from deep_pm.precision_v03 import fit_with_calibration
    n=3
    data=pd.DataFrame(dict(survey=['SDSS','NSC','NSC'],band=['r']*n,
        exposure_id=['SDSS:a','NSC:b','NSC:c'],morphology=[6.,np.nan,.95],
        magnitude=[21.5]*n,magnitude_error=[.1]*n,quality_flags=[0]*n,
        sigma_east_mas=[50.]*n,sigma_north_mas=[50.]*n,
        ra_deg=[30.]*n,dec_deg=[0.]*n,observation_time=[52000.,56000.,57500.]))
    xy=np.zeros((n,2));time=np.array([2001.,2014.,2018.]);sigma=np.full((n,2),70.)
    keys=np.array(data.exposure_id)
    solution={k:(np.zeros(2),np.zeros((48,2)),100) for k in keys}
    def fit(ix):return fit_with_calibration(xy[ix],time[ix],sigma[ix],2015.5,[solution[k] for k in keys[ix]])
    def output(ix):return evaluated_fit(ix,data,xy,time,sigma,keys,solution,[30.,0.],2015.5,model=fit(ix)[0])
    config=dict(reference_epoch=2015.5)
    settings=dict(reference_r_edges=[20,21,22,23],production=dict(match_radius_mas=1000,
        validation_min_recovery=.6,min_nsc_span_years=.6,min_joint_span_years=5,max_axis_pm_error_masyr=10))
    args=(data,xy,time,sigma,np.ones(n,bool),solution,keys,[30.,0.],dict(ra=[29.99,30.01],dec=[-.01,.01]),config,settings,fit,output)
    expected=select_catalogue(*args)
    audit={};actual=select_catalogue(*args,selection_audit=audit)
    def normalized(value):
        if isinstance(value,dict):return {str(k):normalized(v) for k,v in value.items()}
        if isinstance(value,(list,tuple,np.ndarray)):return [normalized(v) for v in value]
        if isinstance(value,np.generic):return value.item()
        return value
    assert normalized(expected)==normalized(actual)
    assert audit['all_r_anchor'][0]['morphology']['state']=='mixed_valid_missing'
    assert audit['all_r_anchor'][0]['status']=='final'
