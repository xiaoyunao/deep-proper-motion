import numpy as np
import pandas as pd
from deep_pm.allband_inventory import summarize


def test_unusual_filter_kept_and_missing_error_not_promoted_by_high_snr():
    data=pd.DataFrame(dict(survey=['NSC']*3,instrument=['c4d']*3,band=['VR','Y','custom'],
        detection_id=['a','b','c'],exposure_id=['NSC:a','NSC:b','NSC:c'],detector_id=[1]*3,
        ra_deg=[10.1]*3,dec_deg=[0.1]*3,observation_time=[57000.]*3,
        sigma_east_mas=[20.,np.nan,20.],sigma_north_mas=[20.]*3,
        magnitude_error=[.01]*3,quality_flags=[0,0,4]))
    result=summarize(data,{'NSC:a':None},dict(ra=[10.,11.],dec=[0.,1.]))
    assert result['actual_nsc_bands']==['VR','Y','custom']
    assert result['unexpected_nsc_band_labels']==['custom']
    assert result['instrument_band']['NSC:c4d:VR']['candidate_reference_supported_count']==1
    assert result['instrument_band']['NSC:c4d:Y']['snr_flags_position_candidate_count']==0
    assert result['instrument_band']['NSC:c4d:custom']['sextractor_fatal_count']==1
