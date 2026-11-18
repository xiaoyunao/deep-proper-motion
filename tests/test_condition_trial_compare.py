import unittest
from deep_pm.condition_trial_compare import join_controls
from deep_pm.independent_runs import ranked_first_batch
from deep_pm.continuous_summary import estimate

class TrialComparisonTest(unittest.TestCase):
    def test_published_comparison_requires_same_detections_and_baseline(self):
        trial=[dict(r_detection_ids=['b','a'],magnitude=22.,values={'W0_r':[[1.,2.],[3.,4.]]})]
        baseline=[dict(detection_ids=['a','b'],magnitude=22.,motion={'all_SDSS':[1.,2.]})]
        self.assertEqual(len(join_controls(trial,baseline)),1)
        trial[0]['r_detection_ids']=['a','c']
        with self.assertRaises(ValueError):join_controls(trial,baseline)

    def test_first_batch_prioritizes_new_runs_without_any_motion_metric(self):
        common=dict(box='a',selected_for_reference_precheck=True,joint_timed_fraction=.95,
                    spectroscopic_qso_21_23_count=50)
        rows=[dict(common,name='old',new_sdss_runs=0,shared_prior_sdss_runs=1),
              dict(common,name='new',new_sdss_runs=1,shared_prior_sdss_runs=0)]
        self.assertEqual(ranked_first_batch(rows)[0]['name'],'new')

    def test_new_model_gain_is_computed_for_the_requested_candidate(self):
        rows=[dict(exposures=[f'SDSS:{i%3}',f'NSC:{i%4}'],nsc_epoch_days=[i%4],
                   motion={'C0':[float(i+1),0.],'multi':[.8*(i+1),0.],'Xiao':[2.*(i+1),0.]}) for i in range(12)]
        result=estimate(rows,('C0','multi','Xiao'),9137,candidate='multi')
        self.assertAlmostEqual(result['paired_radial_gain']['Xiao']['relative_gain'],.6)
        self.assertAlmostEqual(result['paired_radial_gain']['C0']['relative_gain'],.2)
