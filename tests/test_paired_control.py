import unittest
import numpy as np

from deep_pm.paired_control import shared_group_motion


class SharedErrorTest(unittest.TestCase):
    def test_matches_dense_covariance_solution(self):
        time = np.array([2000, 2000.05, 2010, 2010.25, 2017])
        positions = np.array([[5, -20], [6, -19], [10, -12], [15, -8], [23, 0]])
        sigma = np.array([[2, 1], [1, 2], [1, 2], [2, 2], [1, 1]])
        groups = np.array([0, 0, 1, 1, 2])
        design = np.column_stack([np.ones(len(time)), time - time.mean()])
        tau = 1.5
        expected = []
        for axis in range(2):
            cov = np.diag(sigma[:, axis] ** 2) + tau ** 2 * (groups[:, None] == groups)
            u = np.linalg.solve(cov, design)
            expected.append(np.linalg.solve(design.T @ u, u.T @ positions[:, axis])[1])
        np.testing.assert_allclose(shared_group_motion(positions, time, sigma, groups, tau), expected)
