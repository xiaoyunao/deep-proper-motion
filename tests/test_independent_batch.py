import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import numpy as np
import pandas as pd
from astropy.io import fits
from deep_pm.independent_batch import split_cell
from deep_pm.continuous_summary import area
from deep_pm.legacy_regions import extract

class IndependentBatchTest(unittest.TestCase):
    def test_four_small_shards_preserve_the_registered_area(self):
        row=dict(name='a_01',ra_bounds_deg=[132.,134.],dec_bounds_deg=[-3.,-1.])
        parts=split_cell(row)
        self.assertEqual(len(parts),4)
        self.assertAlmostEqual(sum(area(v) for v in parts.values()),area(dict(ra=[132.,134.],dec=[-3.,-1.])))
        self.assertEqual(len(set(tuple(v['ra']+v['dec']) for v in parts.values())),4)

    def test_legacy_fits_keeps_source_flags_native_numbers_and_crop(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);source=root/'input.fits';work=root/'output';work.mkdir()
            columns=[fits.Column(name='objID',format='K',array=[2**55+1,2,3]),
                     fits.Column(name='RA',format='D',array=[10.1,11.9,15.]),
                     fits.Column(name='DEC',format='D',array=[0.,0.,0.]),
                     fits.Column(name='PM_SOURCE',format='5A',array=['DESI','Gaia','DESI'])]
            for key in ('pmRA','pmDEC','pmRA_err','pmDEC_err'):
                columns.append(fits.Column(name=key,format='E',unit='mas/yr',array=[1.,2.,3.]))
            fits.BinTableHDU.from_columns(columns).writeto(source)
            with patch.dict(os.environ,{'SLURM_JOB_ID':'synthetic_unit_test'}):
                extract(work,source,{'a':dict(ra=[10.,12.],dec=[-1.,1.])})
            result=pd.read_csv(work/'xiao_a.csv',dtype={'objID':str})
            self.assertEqual(list(result.objID),[str(2**55+1),'2'])
            self.assertEqual(list(result.PM_SOURCE),['DESI','Gaia'])
            np.testing.assert_allclose(result.pmRA,[1.,2.])
