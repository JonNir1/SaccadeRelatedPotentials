import unittest

import numpy as np
from mne import create_info, io

import mne_scripts.helpers.event_helpers as evh


class TestEventHelpers(unittest.TestCase):

    def test_extract_events(self):
        data = np.zeros((2, 1000))
        info = create_info(['EEG 001', 'STI 014'], sfreq=1000, ch_types=['eeg', 'stim'])
        raw = io.RawArray(data, info, verbose=False)
        raw._data[1, [100, 200, 300]] = [1, 2, 3]
        events = evh.extract_events(raw, channel='STI 014')
        self.assertEqual(events.shape[1], 3)
        self.assertTrue(np.all(events[:, 2] > 0))
        with self.assertRaises(ValueError):
            evh.extract_events(raw, channel='STI_UNKNOWN')