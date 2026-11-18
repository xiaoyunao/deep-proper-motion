import unittest
import numpy as np
import pandas as pd
from deep_pm.condition_error_trial import disjoint_pairs, learn_errors, split_reference


class ErrorTrialTest(unittest.TestCase):
    def test_disjoint_pairs_do_not_treat_repeated_group_or_night_as_independent(self):
        t = np.array([2001.,2001.+.1/365.25,2002.,2003.,2004.])
        groups = np.array(['a','b','a','c','d'])
        self.assertEqual(disjoint_pairs(np.arange(5),t,groups),[(0,3)])
        self.assertEqual(split_reference('physical galaxy',9137),split_reference('physical galaxy',9137))

    def test_error_learning_excludes_c0_training_and_validation_galaxies(self):
        training = next(str(i) for i in range(100) if split_reference(str(i),9137))
        validation = next(str(i) for i in range(100) if not split_reference(str(i),9137))
        refs = pd.DataFrame(dict(reference=[training]*2+[validation]*2+['calibration']*2,
            detection_id=list('abcdef'),is_holdout=[True]*4+[False]*2))
        data = pd.DataFrame(dict(detection_id=list('abcdef'),survey='SDSS',band='r',
                                  sigma_east_mas=40.,sigma_north_mas=40.))
        coord = np.array([[0,0],[10,20],[0,0],[100,200],[0,0],[10000,20000]],float)
        t = np.arange(6,dtype=float)+2000
        sigma = np.hypot(np.full((6,2),40.),50.)
        revised,models,held = learn_errors(refs,data,coord,t,sigma,np.array(list('abcdef')),9137,50)
        self.assertEqual(models['SDSS:r']['pair_count'],1)
        self.assertEqual(models['SDSS:r']['reference_count'],1)
        self.assertEqual(len(held),4)
        self.assertFalse(models['SDSS:r']['enabled_for_trial_only'])
        np.testing.assert_allclose(revised,sigma)

    def test_held_year_detections_cannot_change_the_training_prediction(self):
        from deep_pm.condition_holdout import fixed_year_prediction
        years = np.array([2000.,2001.,2014.3,2015.3,2016.3,2017.3])
        nsc = np.array([False,False,True,True,True,True])
        names = np.array(['SDSS:1','SDSS:2','NSC:1','NSC:2','NSC:3','NSC:4'])
        data = pd.DataFrame(dict(exposure_id=names,detection_id=names,survey=np.where(nsc,'NSC','SDSS'),morphology=.9,magnitude=22.5))
        keys = names
        sigma = np.full((6,2),50.)
        xy = (years-2015.5)[:,None]*np.array([4.,2.])
        offsets = {k:(np.zeros(2),np.zeros((48,2)),40) for k in keys}
        prod = dict(match_radius_mas=1000,min_nsc_span_years=.6,min_joint_span_years=5,
                    validation_min_recovery=.6,max_axis_pm_error_masyr=10,competitor_margin_mas=300)
        settings = dict(production=prod,reference_r_edges=[20,21,22,23])
        config = dict(reference_epoch=2015.5)
        geometry = dict(ra=[349.,351.],dec=[-1.,1.])
        keep = np.ones(6,bool)
        args = (years,sigma,sigma,keys,nsc,keep,keep,offsets,settings,config,geometry,(350.,0.))
        summary,pairs = fixed_year_prediction(data,xy,*args)
        self.assertEqual(summary['status'],'completed')
        self.assertEqual(summary['year'],2015)
        changed = xy.copy(); changed[3,0] += 100.
        _,again = fixed_year_prediction(data,changed,*args)
        self.assertEqual(pairs[0]['held_detection_id'],again[0]['held_detection_id'])
        for method in ('W0_r','W2_r','W0_multi','W2_multi'):
            shift = again[0]['values'][method]['residual_mas']-pairs[0]['values'][method]['residual_mas']
            np.testing.assert_allclose(shift,[100.,0.],atol=1e-8)

    def test_shared_position_zero_cancels_between_training_and_held_prediction(self):
        from deep_pm.astrometry import fit_motion
        from deep_pm.condition_holdout import held_prediction_variance
        t = np.array([2000.,2001.,2015.,2017.])
        sigma = np.full((4,2),50.)
        model = fit_motion(np.zeros((4,2)),t,sigma,2015.5)
        draws = np.random.default_rng(42).normal(size=(48,2))*100
        common = (np.zeros(2),draws,50)
        empty = (np.zeros(2),np.zeros((48,2)),50)
        observed_sigma = np.array([50.,50.])
        a = held_prediction_variance(model,t,sigma,[common]*4,common,observed_sigma,2016.)
        b = held_prediction_variance(model,t,sigma,[empty]*4,empty,observed_sigma,2016.)
        np.testing.assert_allclose(a,b)

    def test_many_pairs_of_one_galaxy_do_not_meet_independent_training_support(self):
        reference=next(str(i) for i in range(100) if split_reference(str(i),9137))
        ids=[str(i) for i in range(120)]
        refs=pd.DataFrame(dict(reference=reference,detection_id=ids,is_holdout=True))
        data=pd.DataFrame(dict(detection_id=ids,survey='NSC',instrument='c4d',band='r',
                               sigma_east_mas=40.,sigma_north_mas=40.))
        t=np.arange(120)/365.25+2000
        sigma=np.hypot(np.full((120,2),40.),50.)
        revised,models,_=learn_errors(refs,data,np.zeros((120,2)),t,sigma,np.array(ids),9137,50)
        model=models['NSC:c4d:r']
        self.assertGreaterEqual(model['pair_count'],50)
        self.assertEqual(model['reference_count'],1)
        self.assertFalse(model['enabled_for_trial_only'])
        np.testing.assert_allclose(revised,sigma)
