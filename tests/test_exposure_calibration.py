import unittest
import numpy as np
from deep_pm.zero_point import fit_offsets
from deep_pm.exposure_catalogues import inside
from deep_pm.exposure_partition import owner, storage_pixels


class ExposureCalibrationTest(unittest.TestCase):
    def test_hierarchy_recovers_detector_difference_and_virtual_parent(self):
        labels = ["SDSS:run|1","SDSS:run|2","NSC:visit|1","NSC:visit|2"]
        xyz = np.tile([[100.,-80.],[-100.,80.],[40.,0.],[-40.,0.]], (28,1))
        refs = np.repeat(np.arange(28),4)
        offsets = fit_offsets(refs, xyz, np.full_like(xyz,40), np.tile(labels,28),
                              5,6,914,parents={k:k.split("|")[0] for k in labels})
        self.assertIn("NSC:visit",offsets)
        self.assertGreater(offsets[labels[0]][0][0]-offsets[labels[1]][0][0],190)
        self.assertEqual(offsets["NSC:visit"][1].shape,(6,2))

    def test_half_open_rectangles_assign_shared_edges_once(self):
        a={"ra":[132,134],"dec":[-1,1]}
        b={"ra":[134,136],"dec":[-1,1]}
        self.assertFalse(inside(134,0,a))
        self.assertTrue(inside(134,0,b))
        self.assertTrue(inside(134,0,a,.06))

    def test_ring_storage_includes_both_sides_of_ra_wrap(self):
        box={"ra":[355,365],"dec":[-5,5]}
        ra=np.array([355,359.9,0,4.9999,5])
        dec=np.array([-5,4.9999,0,-4.9999,0])
        pix=owner(ra,dec,box,128)
        self.assertTrue(set(pix[:4]) <= set(storage_pixels(box,128,.06)))
        self.assertEqual(pix[-1],-1)


if __name__=="__main__":
    unittest.main()
