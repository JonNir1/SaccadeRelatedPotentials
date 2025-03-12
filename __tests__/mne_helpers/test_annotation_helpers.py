import unittest

from mne import Annotations

import mne_scripts.helpers.annotation_helpers as annh

class TestAnnotationHelpers(unittest.TestCase):

    def test_merge_annotations(self):
        onsets = [0, 1.0, 3.0, 4.5]
        durations = [1.5, 1.5, 1.0, 1.0]
        descriptions = ['BAD', 'BAD', 'BAD', 'OTHER']
        annotations = Annotations(onset=onsets, duration=durations, description=descriptions)
        self.assertEqual(len(annh._merge_annotations(annotations, merge_within_ms=0)), 3)
        self.assertEqual(len(annh._merge_annotations(annotations, merge_within_ms=1000)), 2)
        self.assertEqual(len(annh._merge_annotations(annotations, merge_within_ms=-1501)), 4)


