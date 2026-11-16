import unittest

import numpy as np
import pandas as pd

from deep_pm.pilot_production import make_tracks, source_funnel
from deep_pm.pilot_validation import track_audit


class TrialProductionTest(unittest.TestCase):
    def test_two_survey_opportunity_without_calibration_is_counted(self):
        data = pd.DataFrame(dict(survey=["NSC","NSC","SDSS"], exposure_id=["NSC:1","NSC:2","SDSS:1"],
                                 detector_id=[1,1,1],magnitude=[22.2,22.2,21.8],
                                 morphology=[.98,.99,6]))
        xy = np.array([[0.,0.],[0.,10.],[10.,0.]])
        t = np.array([2016.,2018.,2003.])
        keep = np.ones(3,dtype=bool)
        tracks = make_tracks(xy, data.exposure_id.to_numpy(), keep, 1000)
        settings = dict(reference_r_edges=[20,21,22,23], production=dict(match_radius_mas=1000,
            validation_min_recovery=.60,min_nsc_span_years=.60,min_joint_span_years=5,
            max_axis_pm_error_masyr=10))
        result, output = source_funnel(data,xy,t,np.full_like(xy,60),keep,tracks,
                                       {},dict(reference_epoch=2015.5),settings,.1)
        self.assertEqual(result["22_to_23"]["fit_eligible"],1)
        self.assertEqual(result["22_to_23"]["insufficient_calibration_coverage"],1)
        self.assertEqual(result["22_to_23"]["final"],0)
        self.assertFalse(output)

    def test_mixed_known_members_report_contamination(self):
        nsc = np.array([True, False, True, False])
        tracks = [np.arange(4)]
        result, stats = track_audit(tracks,{0:(None,None,None,"lt20",np.arange(4),4)},
            {1:np.array([0,1]),2:np.array([2,3])}, np.array([1,1,2,2]),nsc,.6)
        self.assertEqual(stats["final_mixed_known_tracks"]["n"],1)
        self.assertEqual(stats["final_wrong_known_members"]["n"],2)
        self.assertFalse(result)
