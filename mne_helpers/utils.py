import warnings
from typing import Sequence, Union, List, Optional
from numbers import Number

import numpy as np
import mne

MILLISECONDS_IN_SECOND = 1000
_MIN_FREQ_WARN_THRESHOLD, _MAX_FREQ_WARN_THRESHOLD_WITH_EOG, _MAX_FREQ_WARN_THRESHOLD_NO_EOG = 0.5, 100, 30


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


def resample(raw: mne.io.Raw, resample_freq: float, inplace: bool = False) -> mne.io.Raw:
    """ Resample the data to a new frequency. """
    if not isinstance(resample_freq, Number):
        raise TypeError("`resample_freq` must be a number")
    if resample_freq <= 0:
        raise ValueError("`resample_freq` must be positive")
    new_raw = raw if inplace else raw.copy()
    events = extract_events(new_raw, channel='all')
    return new_raw.resample(sfreq=float(resample_freq), events=events, verbose=False)


def set_reference(raw: mne.io.Raw, ref_channel: Optional[str] = "average", include_eog: bool = True) -> mne.io.Raw:
    """
    Re-reference the EEG data to a new reference channel.
    If `include_eog` is True, also re-references EOG channels to the same reference.
    NOTE: if `include_eog` is True and `ref_channel` is "average", the EOG channels will be included in the average.
    """
    new_raw = raw.copy()
    if not include_eog:
        new_raw.set_eeg_reference(ref_channels=ref_channel, ch_type='eeg', projection=False, verbose=False)
        return new_raw
    # re-reference EEG and EOG channels together: convert EOG to EEG, re-reference, then convert back to EOG
    is_eog_channel = np.array(new_raw.get_channel_types()) == 'eog'
    eog_channel_names = (np.array(new_raw.ch_names)[is_eog_channel]).tolist()
    new_raw.set_channel_types(mapping={ch: 'eeg' for ch in eog_channel_names})
    new_raw.set_eeg_reference(ref_channels=ref_channel, ch_type='eeg', projection=False, verbose=False)
    new_raw.set_channel_types(mapping={ch: 'eog' for ch in eog_channel_names})
    return new_raw


def apply_notch_filter(
        raw: mne.io.Raw,
        freq: float,
        multiplications: int = 5,
        include_eog: bool = True,
        inplace: bool = False,
) -> mne.io.Raw:
    """ Applies a notch filter to the given raw data. """
    assert multiplications > 0, "multiplications must be positive"
    new_raw = raw if inplace else raw.copy()
    channel_types = ["eeg", "eog"] if include_eog else ["eeg"]
    freqs = np.arange(freq, 1 + freq * multiplications, freq).tolist()
    new_raw.notch_filter(freqs=freqs, picks=channel_types)
    return new_raw


def apply_highpass_filter(
        raw: mne.io.Raw,
        min_freq: float,
        include_eog: bool = True,
        inplace: bool = False,
        suppress_warnings: bool = False,
) -> mne.io.Raw:
    assert min_freq > 0, "min_freq must be positive"
    if not suppress_warnings and min_freq > _MIN_FREQ_WARN_THRESHOLD:
        warnings.warn(
            f"High-pass filter of {min_freq}Hz is unusually high. " +
            "Consider setting the cutoff below {_MIN_FREQ_WARN_THRESHOLD}Hz.",
            UserWarning
        )
    new_raw = raw if inplace else raw.copy()
    channel_types = ["eeg", "eog"] if include_eog else ["eeg"]
    new_raw.filter(l_freq=min_freq, h_freq=None, picks=channel_types)
    return new_raw


def apply_lowpass_filter(
        raw: mne.io.Raw,
        max_freq: float,
        include_eog: bool = True,
        inplace: bool = False,
        suppress_warnings: bool = False,
) -> mne.io.Raw:
    assert max_freq > 0, "max_freq must be positive"
    if not suppress_warnings:
        if include_eog and max_freq < _MAX_FREQ_WARN_THRESHOLD_WITH_EOG:
            warnings.warn(
                f"Low-pass filter of {max_freq}Hz is unusually low for EOG data. " +
                f"Consider setting the cutoff above {_MAX_FREQ_WARN_THRESHOLD_WITH_EOG}Hz.",
                UserWarning
            )
        elif not include_eog and max_freq < _MAX_FREQ_WARN_THRESHOLD_NO_EOG:
            warnings.warn(
                f"Low-pass filter of {max_freq}Hz is unusually low. " +
                f"Consider setting the cutoff above {_MAX_FREQ_WARN_THRESHOLD_NO_EOG}Hz.",
                UserWarning
            )
        else:
            pass
    new_raw = raw if inplace else raw.copy()
    channel_types = ["eeg", "eog"] if include_eog else ["eeg"]
    new_raw.filter(l_freq=None, h_freq=max_freq, picks=channel_types)
    return new_raw


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
        channel = channel.strip()
        if channel.lower() == "all":
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

        # add the last annotation
        merged_anns.append((prev_start, prev_end - prev_start, desc))

    new_annots = mne.Annotations(
        onset=np.array([ann[0] for ann in merged_anns]),
        duration=np.array([ann[1] for ann in merged_anns]),
        description=np.array([ann[2] for ann in merged_anns]),
        orig_time=annotations.orig_time,
    )
    return new_annots
