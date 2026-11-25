from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from deep_pm.nsc_cohort import field_resample, ranking, survey_overlap
from deep_pm.nsc_comparison import absolute68, comparison


SETTINGS = dict(seed=321, bootstrap_draws=100, min_publish_count=3, tail_masyr=5,
                min_candidates_with_sdss=8)


class SameSourcePrecisionTest(unittest.TestCase):
    def test_radial_quantile_uses_per_object_vector(self):
        np.testing.assert_allclose(absolute68(np.array([[3,4],[4,3]])), [3.68,3.68,5])

    def test_paired_ratio_of_scaled_motions(self):
        rows = [dict(M3=(np.array([j/2,j/2]), [1,1]),
                     published_nsc=(np.array([j,j]),[1,1])) for j in range(1,7)]
        result = comparison(rows, SETTINGS)
        np.testing.assert_allclose(result["paired"]["M3_vs_published_nsc"]["fraction95"], .5)
        block = field_resample([rows[:3],rows[3:]], SETTINGS)
        np.testing.assert_allclose(block["fraction95"], .5)
        self.assertEqual(comparison(rows[:2], SETTINGS), {"n":2})

    def test_ranking_excludes_low_coverage_and_prioritizes_faint(self):
        stats = {k:dict(candidates_with_both=n,faint_candidates_with_both=f)
                 for k,n,f in (("a",7,7),("b",15,2),("c",10,4))}
        self.assertEqual([k for k,v in ranking(stats, SETTINGS)],["c","b"])


    def test_fields_with_common_runs_are_sampled_together(self):
        with TemporaryDirectory() as root:
            directories = [Path(root)/name for name in ("a","b","c","d")]
            for d, exposure in zip(directories,["SDSS:1","SDSS:1","SDSS:2","SDSS:3"]):
                d.mkdir()
                (d/"detections.csv").write_text(f"survey,exposure_id\nSDSS,{exposure}\n")
            for d in directories[1:3]:
                with (d/"detections.csv").open("a") as f:
                    f.write("NSC,NSC:shared\n")
            self.assertEqual(survey_overlap(directories)[0],[("a","b","c"),("d",)])


    def test_chunked_draws_preserve_full_rng_and_quantiles(self):
        random=np.random.default_rng(79)
        for n in (101,104):
            first=random.normal(size=(n,2))
            second=random.normal(size=(n,2))*2
            rows=[dict(M3=(a,[2.,3.]),published_nsc=(b,[4.,5.]))
                  for a,b in zip(first,second)]
            settings=dict(SETTINGS,bootstrap_draws=37,bootstrap_absolute_zero=True)
            ids=np.random.default_rng(settings['seed']).integers(0,n,(37,n))
            a,b=[absolute68(v[ids],axis=1) for v in (first,second)]
            meda,medb=[np.median(v[ids],axis=1) for v in (first,second)]
            for chunk in (1,7,16,64):
                result=comparison(rows,dict(settings,bootstrap_draw_chunk=chunk))
                pair=result['paired']['M3_vs_published_nsc']
                np.testing.assert_array_equal(pair['absolute68_delta95'],np.quantile(a-b,[.025,.975],axis=0).T)
                np.testing.assert_array_equal(pair['median_bias_delta95'],np.quantile(meda-medb,[.025,.975],axis=0).T)
                np.testing.assert_array_equal(result['methods']['M3']['median_bias95'],np.quantile(meda,[.025,.975],axis=0).T)
                np.testing.assert_array_equal(np.array(pair['fraction95']),np.quantile(1-a/b,[.025,.975],axis=0).T)
