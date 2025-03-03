from typing import Optional, Union, Set

import numpy as np
import pandas as pd
import mne


def eog_blink_annotations(
        raw_data: mne.io.Raw,
        ms_before: int = 25,
        ms_after: int = 25,
) -> mne.Annotations:
    """
    Generate MNE annotations for blinks based on EOG data.

    :param raw_data: MNE Raw object containing the EOG data.
    :param ms_before: duration (in milliseconds) before the blink onset to include in the annotation.
    :param ms_after: duration (in milliseconds) after the blink onset to include in the annotation.

    :return: Blink annotations for MNE.
    """
    assert ms_before >= 0, "ms_before must be a non-negative integer"
    assert ms_after >= 0, "ms_after must be a non-negative integer"
    sfreq = raw_data.info['sfreq']
    samples_before = int(ms_before * sfreq / 1000)
    samples_after = int(ms_after * sfreq / 1000)
    mne_eog_events = mne.preprocessing.find_eog_events(raw_data, verbose=False)

    # Detect Onsets
    onsets = mne_eog_events[:, 0] - samples_before
    onsets[onsets < 0] = 0
    onsets = onsets / sfreq

    # Compute Durations
    durations = np.full_like(onsets, samples_before + samples_after)
    durations[durations + onsets > raw_data.n_times] = raw_data.n_times
    durations = durations / sfreq

    # Create and Return MNE Annotations
    blink_annotations = mne.Annotations(onset=onsets, duration=durations, description=['blink/eog'] * len(onsets))
    return blink_annotations


def eyetracking_blink_annotations(
        raw_data: mne.io.Raw,
        et_channel: str,
        blink_codes: Union[int, Set[int]],
        ms_before: int = 25,
        ms_after: int = 25,
) -> mne.Annotations:
    """
    Generate MNE annotations for blinks based on eye-tracking events.

    :param raw_data: MNE Raw object containing the eye-tracking data.
    :param et_channel: Name of the eye-tracking channel in the raw data.
    :param blink_codes: Event code(s) for blinks.
    :param ms_before: duration (in milliseconds) before the blink onset to include in the annotation.
    :param ms_after: duration (in milliseconds) after the blink offset to include in the annotation.

    :return: Blink annotations for MNE.
    """
    assert isinstance(et_channel, str) and len(et_channel) > 0, "et_channel must be a non-empty string"
    assert isinstance(blink_codes, (int, set)), "blink_codes must be an int or a set of ints"
    assert ms_before >= 0, "ms_before must be a non-negative integer"
    assert ms_after >= 0, "ms_after must be a non-negative integer"
    blink_codes = blink_codes if isinstance(blink_codes, set) else {blink_codes}  # Convert to set if int
    sfreq = raw_data.info['sfreq']
    samples_before = int(ms_before * sfreq / 1000)
    samples_after = int(ms_after * sfreq / 1000)

    # Find blink onsets and offsets
    et_events = mne.find_events(raw_data, stim_channel=et_channel, output='onset', shortest_event=1, consecutive=True)
    blink_mask = np.isin(et_events[:, 2], list(blink_codes))
    blink_onsets = et_events[blink_mask, 0]
    blink_offsets = np.zeros_like(blink_onsets)
    for i, blink_on in enumerate(blink_onsets):
        next_event_idx = np.searchsorted(et_events[:, 0], blink_on, side="right")
        if next_event_idx < len(et_events):
            blink_offsets[i] = et_events[next_event_idx, 0]
        else:
            blink_offsets[i] = et_events[-1, 0] + 1  # Set to last sample + 1
    assert len(blink_offsets) == len(blink_onsets), "Mismatched lengths for blink onsets and offsets"
    assert np.greater_equal(blink_offsets, blink_onsets).all(), "Blink offsets must be >= blink onsets"

    # calculate durations and convert to seconds
    adjusted_onsets = blink_onsets - samples_before  # shift onset backward
    adjusted_onsets[adjusted_onsets < 0] = 0
    adjusted_onsets = adjusted_onsets / sfreq  # convert to seconds for MNE

    blink_durations = blink_offsets - blink_onsets  # duration in samples
    blink_durations[-1] = max(blink_durations[-1], 1)  # Ensure last blink has at least 1 sample
    adjusted_durations = blink_durations + samples_after  # extend duration
    adjusted_durations[adjusted_durations > raw_data.n_times] = raw_data.n_times
    adjusted_durations = adjusted_durations / sfreq  # convert to seconds for MNE

    # Create and Return MNE Annotations
    blink_annotations = mne.Annotations(onset=adjusted_onsets, duration=adjusted_durations, description=['blink/et'] * len(adjusted_onsets))
    return blink_annotations
