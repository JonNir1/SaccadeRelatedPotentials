import unittest

import numpy as np

from utils.calc_utils import *


class TestEventUtils(unittest.TestCase):

    _MILLISECONDS_PER_SECOND = 1000

    def test_calculate_sampling_rate(self):
        t = np.arange(10)
        self.assertEqual(calculate_sampling_rate(t), self._MILLISECONDS_PER_SECOND)
        t = np.arange(0, 11, 2)
        self.assertEqual(calculate_sampling_rate(t), self._MILLISECONDS_PER_SECOND / 2)
        t = np.arange(10) * self._MILLISECONDS_PER_SECOND
        self.assertEqual(calculate_sampling_rate(t), 1)
        t = np.setdiff1d(np.arange(10), [2, 5, 8])
        self.assertEqual(calculate_sampling_rate(t), self._MILLISECONDS_PER_SECOND / 1.5)

