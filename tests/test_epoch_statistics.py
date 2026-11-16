from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from astropy.io import fits
import numpy as np

from deep_pm.footprints import sdss_visits


class SDSSFootprintTests(unittest.TestCase):
    def test_fits_string_rerun_and_vectorized_geometry(self):
        columns = [fits.Column(name="RERUN",format="4A",array=["301","157"])]
        fields = {
            "RUN": [22,23], "RA": [0.,0.], "DEC": [0.,0.],
            "TAI": [5e9,5e9], "NODE": [0.,0.], "INCL": [0.,0.],
            "MU_START": [359.95,359.95], "MU_END": [.05,.05],
            "NU_START": [-.02,-.02], "NU_END": [.02,.02],
        }
        for key, value in fields.items():
            columns.append(fits.Column(name=key,format="D",array=value))
        with TemporaryDirectory() as directory:
            path = Path(directory)/"window.fits"
            fits.BinTableHDU.from_columns(columns).writeto(path)
            vector, = sdss_visits(path,(0.,0.),.25,rerun=301,vectorized=True)
            scalar, = sdss_visits(path,(0.,0.),.25,rerun=301)
            sky = np.array([[359.97,0.],[.03,.01],[.06,0.],[.01,.03]])
            self.assertEqual(vector.exposure,"SDSS:22")
            np.testing.assert_array_equal(vector.coverage(None,sky),[True,True,False,False])
            np.testing.assert_array_equal([scalar.coverage(None,p) for p in sky],vector.coverage(None,sky))
