import unittest

import healpy as hp
import numpy as np

from deep_pm.legacy_subset import find_ring_boundary


class RangeBoundaryTest(unittest.TestCase):
    def test_ring_boundaries_are_verified(self):
        pixels = np.repeat([2, 4, 16, 100], [5, 150, 1, 120])
        rows = np.rec.fromarrays(hp.pix2ang(128, pixels, lonlat=True), names="RA,DEC")
        for key in (0, 2, 4, 5, 16, 17, 100, 101):
            self.assertEqual(find_ring_boundary(len(rows), key, lambda i, n: rows[i:i+n]),
                             np.searchsorted(pixels, key))

    def test_detected_ring_inversion_rejects_range(self):
        rows = np.rec.fromarrays(hp.pix2ang(128, [20, 10], lonlat=True), names="RA,DEC")
        with self.assertRaisesRegex(ValueError, "locally inconsistent"):
            find_ring_boundary(len(rows), 15, lambda i, n: rows[i:i+n])
