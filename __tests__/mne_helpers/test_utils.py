import unittest

import numpy as np
from mne import create_info, io

import mne_scripts.helpers.utils as u


class TestUtils(unittest.TestCase):

    def test_to_vector(self):
        lst = [1, 2, 3]
        exp = np.array(lst)
        self.assertTrue(np.array_equal(u.to_vector(lst), exp, equal_nan=True))
        self.assertRaises(TypeError, u.to_vector, np.array([lst, lst]))
        self.assertRaises(TypeError, u.to_vector, np.array([lst, lst, lst]))

    def test_calculate_sampling_rate(self):
        t = np.arange(10)
        self.assertEqual(u.calculate_sampling_rate(t), u.MILLISECONDS_IN_SECOND)
        t = np.arange(0, 11, 2)
        self.assertEqual(u.calculate_sampling_rate(t), u.MILLISECONDS_IN_SECOND / 2)
        t = np.arange(10) * u.MILLISECONDS_IN_SECOND
        self.assertEqual(u.calculate_sampling_rate(t), 1)
        t = np.setdiff1d(np.arange(10), [2, 5, 8])
        self.assertEqual(u.calculate_sampling_rate(t), u.MILLISECONDS_IN_SECOND / 1.5)

    def test_milliseconds_to_samples(self):
        self.assertEqual(u.milliseconds_to_samples(1000, 500), 500)
        self.assertEqual(u.milliseconds_to_samples(0, 500), 0)
        self.assertEqual(u.milliseconds_to_samples(250, 1000), 250)
        with self.assertRaises(AssertionError):
            u.milliseconds_to_samples(-10, 500)
        with self.assertRaises(AssertionError):
            u.milliseconds_to_samples(10, 0)
