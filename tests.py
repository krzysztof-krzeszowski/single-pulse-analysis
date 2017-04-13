#!/usr/bin/env python

import functions as fn
import numpy as np
import unittest


class Test(unittest.TestCase):
    def test_get_baseline_single(self):
        d = np.array([4, 2, 5, 8, 4, 1, 2, 7, 9, 0, 7, 5, 2, 3, 7, 9, 2])
        np.testing.assert_almost_equal(fn.get_baseline_single(d, position=4, width=5), 4.6, decimal=4)

    def test_get_baseline_multiple(self):
        d = np.array([
            [9, 4, 7, 7, 4, 4, 4, 6, 0, 5],
            [6, 4, 4, 7, 8, 3, 6, 6, 8, 9],
            [9, 6, 2, 1, 0, 3, 9, 9, 2, 8]
        ])
        positions = np.array([2, 5, 1])
        ans = np.array([6, 5, 3])
        np.testing.assert_array_almost_equal(fn.get_baselines(d, position=positions, width=3), ans, decimal=4)
        
    def test_subtract_baseline_multiple(self):
        d = np.array([
            [9, 4, 7, 7, 4, 4, 4, 6, 0, 5],
            [6, 4, 4, 7, 8, 3, 6, 6, 8, 9],
            [9, 6, 2, 1, 0, 3, 9, 9, 2, 8]
        ])
        ans = np.array([
            [ 3., -2.,  1.,  1., -2., -2., -2.,  0., -6., -1.],
            [-0.3333, -2.3333, -2.3333,  0.6667,  1.6667, -3.3333, -0.3333, -0.3333,  1.6667,  2.6667],
            [ 8.,  5.,  1.,  0., -1.,  2.,  8.,  8.,  1.,  7.]
        ])
        np.testing.assert_array_almost_equal(fn.subtract_baseline(d, position=2, width=3), ans, decimal=4)
        
    def test_subtract_baseline_single(self):
        d = np.array([4, 2, 5, 8, 4, 1, 2, 7, 9, 0, 7, 5, 2, 3, 7, 9, 2])
        ans = np.array([-0.6, -2.6,  0.4,  3.4, -0.6, -3.6, -2.6,  2.4,  4.4, -4.6,  2.4,
        0.4, -2.6, -1.6,  2.4,  4.4, -2.6])
        np.testing.assert_array_almost_equal(fn.subtract_baseline_single(d, position=4, width=5), ans)
    
    def test_mean_profile(self):
        d = np.array([[1, 2, 3, 4, 5], [3, 4, 5, 6, 7], [5, 6, 7, 8, 9]])
        np.testing.assert_array_equal(fn.get_mean_profile(d), np.array([3, 4, 5, 6, 7]))

    def test_running_sd_single(self):
        d = np.array([4, 2, 5, 8, 4, 1, 2, 7, 9, 0, 7, 5, 2, 3, 7, 9, 2])
        np.testing.assert_array_almost_equal(fn.get_sd_from_pulse(d, width=3, step=2), np.array([[1.2472, 0], [3.8586, 8]]), decimal=4)

    def test_running_sd_pulses(self):
        d = np.array([
            [9, 4, 7, 7, 4, 4, 4, 6, 0, 5],
            [6, 4, 4, 7, 8, 3, 6, 6, 8, 9],
            [9, 6, 2, 1, 0, 3, 9, 9, 2, 8]
        ])
        ans = np.array([
            [[1.4697, 2], [1.9596, 4]],
            [[1.6000, 0], [1.8547, 2]],
            [[3.1623, 2], [3.7202, 4]],
        ])

        np.testing.assert_array_almost_equal(fn.get_sd_from_pulses(d, width=5, step=2), ans, decimal=4)

if __name__ == '__main__':
    unittest.main()
