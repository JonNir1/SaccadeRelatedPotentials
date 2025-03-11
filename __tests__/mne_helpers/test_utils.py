import unittest

import numpy as np
import mne
from mne import Annotations, create_info, io

import mne_helpers.utils as u


class TestUtils(unittest.TestCase):
    _MILLISECONDS_PER_SECOND = 1000

    def test_to_vector(self):
        lst = [1, 2, 3]
        exp = np.array(lst)
        self.assertTrue(np.array_equal(u.to_vector(lst), exp, equal_nan=True))
        self.assertRaises(TypeError, u.to_vector, np.array([lst, lst]))
        self.assertRaises(TypeError, u.to_vector, np.array([lst, lst, lst]))

    def test_calculate_sampling_rate(self):
        t = np.arange(10)
        self.assertEqual(u.calculate_sampling_rate(t), self._MILLISECONDS_PER_SECOND)
        t = np.arange(0, 11, 2)
        self.assertEqual(u.calculate_sampling_rate(t), self._MILLISECONDS_PER_SECOND / 2)
        t = np.arange(10) * self._MILLISECONDS_PER_SECOND
        self.assertEqual(u.calculate_sampling_rate(t), 1)
        t = np.setdiff1d(np.arange(10), [2, 5, 8])
        self.assertEqual(u.calculate_sampling_rate(t), self._MILLISECONDS_PER_SECOND / 1.5)

    def test_milliseconds_to_samples(self):
        self.assertEqual(u.milliseconds_to_samples(1000, 500), 500)
        self.assertEqual(u.milliseconds_to_samples(0, 500), 0)
        self.assertEqual(u.milliseconds_to_samples(250, 1000), 250)
        with self.assertRaises(AssertionError):
            u.milliseconds_to_samples(-10, 500)
        with self.assertRaises(AssertionError):
            u.milliseconds_to_samples(10, 0)

    def test_extract_events(self):
        data = np.zeros((2, 1000))
        info = create_info(['EEG 001', 'STI 014'], sfreq=1000, ch_types=['eeg', 'stim'])
        raw = io.RawArray(data, info, verbose=False)
        raw._data[1, [100, 200, 300]] = [1, 2, 3]
        events = u.extract_events(raw, channel='STI 014')
        self.assertEqual(events.shape[1], 3)
        self.assertTrue(np.all(events[:, 2] > 0))
        with self.assertRaises(ValueError):
            u.extract_events(raw, channel='STI_UNKNOWN')

    def test_merge_annotations(self):
        onsets = [0, 1.0, 3.0, 4.5]
        durations = [1.5, 1.5, 1.0, 1.0]
        descriptions = ['BAD', 'BAD', 'BAD', 'OTHER']
        annotations = Annotations(onset=onsets, duration=durations, description=descriptions)
        self.assertEqual(len(u.merge_annotations(annotations, merge_within_ms=0)), 3)
        self.assertEqual(len(u.merge_annotations(annotations, merge_within_ms=1000)), 2)
        self.assertEqual(len(u.merge_annotations(annotations, merge_within_ms=-1501)), 4)