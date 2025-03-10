from typing import Sequence, Union, List

import numpy as np
import mne

MILLISECONDS_IN_SECOND = 1000


def milliseconds_to_samples(ms: float, sfreq: float) -> int:
    """ Converts milliseconds to samples given the sampling frequency. """
    assert ms >= 0, "milliseconds must be non-negative"
    assert sfreq > 0, "sampling frequency must be positive"
    return int(round(ms * sfreq / MILLISECONDS_IN_SECOND, 0))


def to_vector(seq: Sequence) -> np.ndarray:
    arr = np.array(seq)
    if arr.ndim == 1:
        return arr
    if arr.ndim >= 3:
        raise TypeError(f"Cannot convert an array with {arr.ndim} dimensions to a vector.")
    # reached here if array.ndim == 2
    assert arr.ndim == 2
    rows, cols = arr.shape
    if rows == 1:
        return arr[0]
    if cols == 1:
        return arr[:, 0]
    raise TypeError(f"Cannot convert a 2D matrix with {rows} rows and {cols} columns to a vector.")


def calculate_sampling_rate(milliseconds: np.ndarray, decimals: int = None) -> float:
    """
    Calculates the sampling rate of the given timestamps in Hz.
    :param milliseconds: timestamps in milliseconds (floating-point, not integer)
    :param decimals: number of decimal places to round to
    """
    if len(milliseconds) < 2:
        raise ValueError("timestamps must be of length at least 2")
    if decimals is not None and not isinstance(decimals, int):
        raise TypeError("decimals must be an integer")
    if decimals is not None and decimals < 0:
        raise ValueError("decimals must be non-negative")
    ms_per_sec = 1000
    sr = ms_per_sec / np.median(np.diff(milliseconds))
    if not np.isfinite(sr):
        raise RuntimeError("Error calculating sampling rate")
    if decimals is None:
        return float(sr)
    return round(sr, decimals)


def extract_events(
        raw: mne.io.Raw, channel: Union[str, List[str]], output: str = "onset", shortest_event: int = 1
) -> np.ndarray:
    """
    Extract events from one or more stimulus channels in a Raw object.

    :param raw: The MNE Raw object containing stim channel data.
    :param channel: A single stim channel name, a list of names, or "all" to use all stim channels.
    :param output: The type of event boundary to return. One of "onset", "offset", or "step". Default is "onset".
    :param shortest_event: Minimum number of samples for an event to be considered valid. Default is 1.

    :returns: An MNE-style event array (n_events × 3) with sample, previous value, and new value.
    """
    output = output.strip().lower()
    assert output in ["onset", "offset", "step"], "Invalid output type. Must be 'onset', 'offset', or 'step'."
    assert shortest_event > 0, "shortest_event must be non-negative"
    all_channel_names = np.array(raw.ch_names)
    is_stim_channel = np.array(raw.get_channel_types()) == "stim"
    stim_channel_names = (all_channel_names[is_stim_channel]).tolist()
    if isinstance(channel, str):
        channel = channel.lower().strip()
        if channel == "all":
            return mne.find_events(
                raw,
                stim_channel=stim_channel_names,
                output=output,
                shortest_event=shortest_event,
                consecutive=True,
                verbose=False
            )
        channel = [channel]
    unknown_channels = set(channel) - set(stim_channel_names)
    if unknown_channels:
        raise ValueError(f"Unknown stim channel(s): {unknown_channels}")
    return mne.find_events(
        raw,
        stim_channel=channel,
        output=output,
        shortest_event=shortest_event,
        consecutive=True,
        verbose=False
    )


def merge_annotations(annotations: mne.Annotations, merge_within_ms: float = 0.0) -> mne.Annotations:
    """
    Merges annotations if they share the same description and are within `merge_within_ms` seconds of each other.
    Note: negative `merge_within_ms` indicate merging requires overlapping annotations.
    Returns a new `mne.Annotations` object with the merged annotations.
    """
    merged_anns = []
    merge_within_sec = merge_within_ms / MILLISECONDS_IN_SECOND
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
        merged_anns.append((prev_start, prev_end - prev_start, desc))   # add the last annotation
    new_annots = mne.Annotations(
        onset=np.array([ann[0] for ann in merged_anns]),
        duration=np.array([ann[1] for ann in merged_anns]),
        description=np.array([ann[2] for ann in merged_anns]),
        orig_time=annotations.orig_time,
    )
    return new_annots
