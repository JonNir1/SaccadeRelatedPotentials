import unittest

import numpy as np

import mne_helpers.utils as u


class TestUtils(unittest.TestCase):
    _MILLISECONDS_PER_SECOND = 1000

    def test_to_vector(self):
        lst = [1, 2, 3]
        exp = np.array(lst)
        self.assertTrue(np.array_equal(u.to_vector(lst), exp, equal_nan=True))
        self.assertTrue(np.array_equal(u.to_vector(lst), exp, equal_nan=True))
        self.assertTrue(np.array_equal(u.to_vector(lst), exp, equal_nan=True))
        self.assertRaises(TypeError, u.to_vector, np.array([lst, lst]))
        self.assertRaises(TypeError, u.to_vector, np.array([lst, lst, lst]))

    def test_calculate_sampling_rate(self):
        # test basic functionality
        t = np.arange(10)
        self.assertEqual(u.calculate_sampling_rate(t), self._MILLISECONDS_PER_SECOND)
        t = np.arange(0, 11, 2)
        self.assertEqual(u.calculate_sampling_rate(t), self._MILLISECONDS_PER_SECOND / 2)
        t = np.arange(10) * self._MILLISECONDS_PER_SECOND
        self.assertEqual(u.calculate_sampling_rate(t), 1)
        t = np.setdiff1d(np.arange(10), [2, 5, 8])
        self.assertEqual(u.calculate_sampling_rate(t), self._MILLISECONDS_PER_SECOND / 1.5)
