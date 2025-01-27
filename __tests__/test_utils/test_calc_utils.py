import unittest

import numpy as np

from utils.calc_utils import *


class TestEventUtils(unittest.TestCase):

    _MILLISECONDS_PER_SECOND = 1000

    def test_calculate_sampling_rate(self):
        # test basic functionality
        t = np.arange(10)
        self.assertEqual(calculate_sampling_rate(t), self._MILLISECONDS_PER_SECOND)
        t = np.arange(0, 11, 2)
        self.assertEqual(calculate_sampling_rate(t), self._MILLISECONDS_PER_SECOND / 2)
        t = np.arange(10) * self._MILLISECONDS_PER_SECOND
        self.assertEqual(calculate_sampling_rate(t), 1)
        t = np.setdiff1d(np.arange(10), [2, 5, 8])
        self.assertEqual(calculate_sampling_rate(t), self._MILLISECONDS_PER_SECOND / 1.5)

        # test rounding
        t = np.setdiff1d(np.arange(10), [2, 5, 8])
        self.assertEqual(calculate_sampling_rate(t, 0), round(self._MILLISECONDS_PER_SECOND / 1.5, 0))
        self.assertEqual(calculate_sampling_rate(t, 5), round(self._MILLISECONDS_PER_SECOND / 1.5, 5))

        # test exceptions
        with self.assertRaises(ValueError, msg="timestamps must be of length at least 2"):
            calculate_sampling_rate(np.array([1]))
        with self.assertRaises(TypeError, msg="decimals must be an integer"):
            calculate_sampling_rate(np.arange(10), decimals=1.5)
        with self.assertRaises(ValueError, msg="decimals must be non-negative"):
            calculate_sampling_rate(np.arange(10), decimals=-1)

