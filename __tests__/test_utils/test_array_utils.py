import unittest

import numpy as np

import utils.array_utils as au


class TestArrayUtils(unittest.TestCase):

    def test_to_vector(self):
        lst = [1, 2, 3]
        exp = np.array(lst)
        self.assertTrue(np.array_equal(au.to_vector(lst), exp, equal_nan=True))
        self.assertTrue(np.array_equal(au.to_vector(lst), exp, equal_nan=True))
        self.assertTrue(np.array_equal(au.to_vector(lst), exp, equal_nan=True))
        self.assertRaises(TypeError, au.to_vector, np.array([lst, lst]))
        self.assertRaises(TypeError, au.to_vector, np.array([lst, lst, lst]))
