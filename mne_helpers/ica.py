from typing import Optional, Union, Set, Dict

import numpy as np
import mne

import mne_helpers.utils as u


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
        et_blinks = _eyetracking_blink_annotation(
            raw,
            et_channel=et_channel, blink_codes=blink_codes,
            ms_before=0, ms_after=1, merge_within_ms=1
        )
    else:
        et_blinks = mne.Annotations([], [], [])

    # detect blinks from EOG
    if kwargs.get("use_eog_blinks", True):
        eog_blinks = _eog_blink_annotation(
            raw,
            threshold=kwargs.get("eog_blinks_threshold", 400e-6),
            ms_before=0, ms_after=1, merge_within_ms=1
        )
    else:
        eog_blinks = mne.Annotations([], [], [])

    # convert blink annotations to epochs
    blink_annots = eog_blinks + et_blinks   # merge ET and EOG blink annotations
    blink_onsets = blink_annots.onset
    # TODO - continue from here
    return None


def _eyetracking_blink_annotation(
        raw_data: mne.io.Raw,
        et_channel: str,
        blink_codes: Union[int, Set[int]],
        ms_before: int = 25,
        ms_after: int = 25,
        merge_within_ms: float = 0.0,
) -> mne.Annotations:
    """
    Generate MNE annotations for blinks based on eye-tracking events.

    :param raw_data: MNE Raw object containing the eye-tracking data.
    :param et_channel: Name of the eye-tracking channel in the raw data.
    :param blink_codes: Event code(s) for blinks.
    :param ms_before: duration (in milliseconds) before the blink onset to include in the annotation.
    :param ms_after: duration (in milliseconds) after the blink offset to include in the annotation.
    :param merge_within_ms: time window (in milliseconds) within which adjacent annotations are merged.

    :return: Blink annotations for MNE.
    """
    assert isinstance(et_channel, str) and len(et_channel) > 0, "et_channel must be a non-empty string"
    assert isinstance(blink_codes, (int, set)), "blink_codes must be an int or a set of ints"
    assert ms_before >= 0, "ms_before must be a non-negative integer"
    assert ms_after >= 0, "ms_after must be a non-negative integer"
    blink_codes = blink_codes if isinstance(blink_codes, set) else {blink_codes}  # Convert to set if int
    sfreq = raw_data.info['sfreq']
    samples_before = u.milliseconds_to_samples(ms_before, sfreq)
    samples_after = u.milliseconds_to_samples(ms_after, sfreq)

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
    blink_annotations = mne.Annotations(
        onset=adjusted_onsets,
        duration=adjusted_durations,
        description=['blink/et'] * len(adjusted_onsets)
    )
    return u.merge_annotations(blink_annotations, merge_within_ms)


def _eog_blink_annotation(
        raw: mne.io.Raw,
        threshold: Optional[float] = None,
        ms_before: int = 250,
        ms_after: int = 250,
        merge_within_ms: float = 0.0,
) -> mne.Annotations:
    """
    Generate MNE annotations for blinks based on EOG data.

    :param raw: MNE Raw object containing the EOG data.
    :param threshold: threshold for detecting blinks. If None, uses MNE's default threshold.
    :param ms_before: duration (in milliseconds) before the blink onset to include in the annotation.
    :param ms_after: duration (in milliseconds) after the blink onset to include in the annotation.
    :param merge_within_ms: time window (in milliseconds) within which adjacent annotations are merged.

    :return: mne.Annotations object marking the blinks.
    """
    assert threshold is None or threshold > 0, "threshold must be a positive float or None"
    assert ms_before >= 0, "ms_before must be a non-negative integer"
    assert ms_after >= 0, "ms_after must be a non-negative integer"
    sfreq = raw.info['sfreq']
    samples_before = u.milliseconds_to_samples(ms_before, sfreq)
    samples_after = u.milliseconds_to_samples(ms_after, sfreq)
    mne_eog_events = mne.preprocessing.find_eog_events(raw, threshold=threshold, verbose=False)

    # Detect Onset Idxs
    onsets = mne_eog_events[:, 0] - samples_before
    onsets[onsets < 0] = 0

    # Compute Durations in Sample Units
    durations = np.full_like(onsets, samples_before + samples_after)
    durations[durations + onsets > raw.n_times] = raw.n_times

    # Create and Return MNE Annotations
    blink_annotations = mne.Annotations(
        onset=onsets / sfreq,           # convert to seconds
        duration=durations / sfreq,     # convert to seconds
        description=['blink/eog'] * len(onsets),
    )
    return u.merge_annotations(blink_annotations, merge_within_ms)
