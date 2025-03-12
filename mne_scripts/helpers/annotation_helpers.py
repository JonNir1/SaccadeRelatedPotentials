from typing import Optional, Union, Set

import numpy as np
import mne

import mne_scripts.helpers.utils as u


def voltage_jump_annotations(
        raw: mne.io.Raw,
        channel_type: str,
        jump_threshold_volts: float = 2e-4,
        jump_window_ms: int = 100,
        min_channel_ratio: float = 0.5,
        pre_annotation_ms: float = 250,
        post_annotation_ms: float = 250,
        merge_within_ms: float = 0.0,
) -> mne.Annotations:
    """
    Detects abrupt voltage jumps across many channels and returns annotations marking those periods.

    :param raw: The raw EEG/EOG data to analyze.
    :param channel_type: Channel type to process ("eeg" or "eog").
    :param jump_threshold_volts: Voltage difference threshold (in volts) to consider a "jump".
    :param jump_window_ms: Number of milliseconds between samples to compute jump differences.
    :param min_channel_ratio: Minimum fraction of channels that must exceed the threshold to mark a jump.
    :param pre_annotation_ms: Time (ms) to extend the annotation before the detected jump.
    :param post_annotation_ms: Time (ms) to extend the annotation after the detected jump.
    :param merge_within_ms: Time window (ms) within which adjacent annotations of the same kind are merged.

    :return: MNE annotations marking suspected voltage jumps.
    """
    assert jump_threshold_volts > 0, "Voltage threshold should be a positive value."
    assert 0 < min_channel_ratio < 1, "Channel ratio should be a value between 0 and 1."
    assert jump_window_ms > 0, "Time difference should be a positive value."
    assert pre_annotation_ms >= 0, "Annotation-time before should be a non-negative value."
    assert post_annotation_ms >= 0, "Annotation-time after should be a non-negative value."

    # get the data for the specified channel type
    channel_type = channel_type.strip().lower()
    assert channel_type in ['eeg', 'eog'], "Unsuitable channel type. Should be 'eeg' or 'eog'."
    try:
        data = raw.get_data(picks=channel_type)
    except ValueError as e:
        if "could not be interpreted as" in str(e):
            # couldn't find channels of this type, return empty annotations
            return mne.Annotations(onset=[], duration=[], description=[])
        raise e     # re-raise the exception if it's not the one we're expecting

    # calculate volt diffs in jumps of `ms_difference`
    n_channels, n_samples = data.shape
    sample_difference = u.milliseconds_to_samples(jump_window_ms, raw.info['sfreq'])
    assert sample_difference < n_samples, "Time difference too large for the data."
    voltage_diffs = data[:, :(n_samples - sample_difference)] - data[:, sample_difference:]

    # find time points where the voltage difference exceeds the threshold in enough channels
    is_sample_above_threshold = np.abs(voltage_diffs) > jump_threshold_volts
    ratio_channels_above_threshold = np.sum(is_sample_above_threshold, axis=0) / n_channels     # for each time point
    is_jump = ratio_channels_above_threshold > min_channel_ratio

    # create annotations
    onsets = raw.times[np.where(is_jump)[0]] - pre_annotation_ms / u.MILLISECONDS_IN_SECOND
    onsets[onsets < 0] = 0
    durations = np.full_like(onsets, (pre_annotation_ms + post_annotation_ms) / u.MILLISECONDS_IN_SECOND)
    durations[onsets + durations > raw.times[-1]] = raw.times[-1]
    annotations = mne.Annotations(onset=onsets, duration=durations, description=['voltage_jump'] * len(onsets))
    return _merge_annotations(annotations, merge_within_ms)


def eyetracking_blink_annotation(
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
    return _merge_annotations(blink_annotations, merge_within_ms)


def eog_blink_annotation(
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
    return _merge_annotations(blink_annotations, merge_within_ms)


def _merge_annotations(annotations: mne.Annotations, merge_within_ms: float = 0.0) -> mne.Annotations:
    """
    Merges annotations if they share the same description and are within `merge_within_ms` seconds of each other.
    Note: negative `merge_within_ms` indicate merging requires overlapping annotations.
    Returns a new `mne.Annotations` object with the merged annotations.
    """
    merged_anns = []
    merge_within_sec = merge_within_ms / u.MILLISECONDS_IN_SECOND
    descriptions = set(annotations.description)
    for desc in descriptions:
        # extract only the annotations with the same description
        is_desc_mask = annotations.description == desc
        onsets = annotations.onset[is_desc_mask]
        durations = annotations.duration[is_desc_mask]

        # sort by onset time
        sorted_indices = np.argsort(onsets)
        onsets, durations = onsets[sorted_indices], durations[sorted_indices]

        # merge adjacent annotations
        prev_start, prev_end = onsets[0], onsets[0] + durations[0]
        for i in range(1, len(onsets)):
            curr_start, curr_end = onsets[i], onsets[i] + durations[i]
            if curr_start <= prev_end + merge_within_sec:
                # extend the previous annotation
                prev_end = max(prev_end, curr_end)
            else:
                merged_anns.append((prev_start, prev_end - prev_start, desc))
                prev_start, prev_end = curr_start, curr_end

        # add the last annotation
        merged_anns.append((prev_start, prev_end - prev_start, desc))

    new_annots = mne.Annotations(
        onset=np.array([ann[0] for ann in merged_anns]),
        duration=np.array([ann[1] for ann in merged_anns]),
        description=np.array([ann[2] for ann in merged_anns]),
        orig_time=annotations.orig_time,
    )
    return new_annots
