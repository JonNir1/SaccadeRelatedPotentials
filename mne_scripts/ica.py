from typing import Dict

import mne

import mne_scripts.helpers.annotation_helpers as annh


def run_ica(raw: mne.io.Raw, event_dict: Dict[str, int], **kwargs) -> mne.io.Raw:

    return None


def _blink_epochs(raw: mne.io.Raw, **kwargs) -> mne.Epochs:
    # detect blinks from eyetracking
    if kwargs.get("use_eyetracking_blinks", kwargs.get("use_et_blinks", True)):
        et_channel = kwargs.get("et_channel", None)
        if et_channel is None:
            raise ValueError("`et_channel` must be provided when using eyetracking blinks")
        blink_codes = kwargs.get("et_blink_codes", None)
        if blink_codes is None:
            raise ValueError("`et_blink_codes` must be provided when using eyetracking blinks")
        et_blinks = annh.eyetracking_blink_annotation(
            raw, et_channel=et_channel, blink_codes=blink_codes, ms_before=0, ms_after=1, merge_within_ms=1
        )
    else:
        et_blinks = mne.Annotations([], [], [])

    # detect blinks from EOG
    if kwargs.get("use_eog_blinks", True):
        eog_blinks = annh.eog_blink_annotation(
            raw, threshold=kwargs.get("eog_blinks_threshold", 400e-6), ms_before=0, ms_after=1, merge_within_ms=1
        )
    else:
        eog_blinks = mne.Annotations([], [], [])

    # convert blink annotations to epochs
    blink_annots = eog_blinks + et_blinks   # merge ET and EOG blink annotations
    blink_onsets = blink_annots.onset
    # TODO - continue from here
    return None

