#!/usr/bin/env python

import functions as fn
import numpy as np
import unittest


class Test(unittest.TestCase):
    def test_mean_profile(self):
        d = np.array([[1, 2, 3, 4, 5], [3, 4, 5, 6, 7], [5, 6, 7, 8, 9]])
        np.testing.assert_array_equal(fn.get_mean_profile(d), np.array([3, 4, 5, 6, 7]))

    def test_running_sd_single(self):
        d = np.array([4, 2, 5, 8, 4, 1, 2, 7, 9, 0, 7, 5, 2, 3, 7, 9, 2])
        np.testing.assert_array_almost_equal(fn.get_sd_from_pulse(d, width=3), np.array([1.2472, 3.8586]), decimal=4)

    def test_running_sd_pulses(self):
        d = np.array([
            [9, 4, 7, 7, 4, 4, 4, 6, 0, 5],
            [6, 4, 4, 7, 8, 3, 6, 6, 8, 9],
            [9, 6, 2, 1, 0, 3, 9, 9, 2, 8]
        ])
        ans = np.array([
            [1.2649, 2.0396],
            [1.6000, 2.0591],
            [2.0591, 3.8781],
        ])

        np.testing.assert_array_almost_equal(fn.get_sd_from_pulses(d, width=5), ans, decimal=4)

if __name__ == '__main__':
    unittest.main()
