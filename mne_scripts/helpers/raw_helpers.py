import warnings
from typing import Optional, Union, List
from numbers import Number

import numpy as np
import mne

import mne_scripts.helpers.event_helpers as evh

_MIN_FREQ_WARN_THRESHOLD, _MAX_FREQ_WARN_THRESHOLD_WITH_EOG, _MAX_FREQ_WARN_THRESHOLD_NO_EOG = 0.5, 100, 30


def resample(raw: mne.io.Raw, resample_freq: float, inplace: bool = False) -> (mne.io.Raw, np.ndarray):
    """ Resample the data to a new frequency, and return the new raw object and the updated events. """
    if not isinstance(resample_freq, Number):
        raise TypeError("`resample_freq` must be a number")
    if resample_freq <= 0:
        raise ValueError("`resample_freq` must be positive")
    new_raw = raw if inplace else raw.copy()
    events = evh.extract_events(new_raw, channel='all')
    new_raw, new_events = new_raw.resample(sfreq=float(resample_freq), events=events, verbose=False)
    return new_raw, new_events


def set_new_montage(raw: mne.io.Raw, montage: Optional[str] = None, overwrite: bool = False) -> mne.io.Raw:
    """ Set the montage for the data, optionally overwriting the existing montage. """
    new_raw = raw.copy()
    if new_raw.get_montage() is None:
        new_raw.set_montage(montage, on_missing='ignore', verbose=False)
        return new_raw
    if not overwrite:
        return new_raw
    warnings.warn(f"Attempting to set montage '{montage}' on data with existing montage.")
    new_raw.set_montage(montage, on_missing='ignore', verbose=False)
    return new_raw


def set_reference(
        raw: mne.io.Raw, ref_channel: Optional[Union[List[str], str]] = "average", include_eog: bool = True
) -> mne.io.Raw:
    """
    Re-reference the EEG data to a new reference channel.
    If `include_eog` is True, also re-references EOG channels to the same reference.
    NOTE: if `include_eog` is True and `ref_channel` is "average", the EOG channels will be included in the average.
    """
    new_raw = raw.copy()
    if isinstance(ref_channel, str) and ref_channel.lower() != "average":
        ref_channel = [ref_channel]
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


def remap_channels(
        raw: mne.io.Raw,
        eog_channels: Optional[List[str]] = None,
        stim_channels: Optional[List[str]] = None,
        gaze_channels: Optional[List[str]] = None,
        pupil_channels: Optional[List[str]] = None,
        misc_channels: Optional[List[str]] = None,
) -> mne.io.Raw:
    """ Remap channels to new types based on user-provided mappings. """
    new_raw = raw.copy()
    mapping = dict()
    mapping.update({ch: 'eog' for ch in (eog_channels or [])})
    mapping.update({ch: 'stim' for ch in (stim_channels or [])})
    mapping.update({ch: 'eyegaze' for ch in (gaze_channels or [])})
    mapping.update({ch: 'pupil' for ch in (pupil_channels or [])})
    mapping.update({ch: 'misc' for ch in (misc_channels or [])})
    unknown_channels = set(mapping.keys()) - set(raw.ch_names)
    if unknown_channels:
        warnings.warn(f"Ignoring unknown channel(s) in mapping: {unknown_channels}.")
        [mapping.pop(ch) for ch in unknown_channels]
    new_raw.set_channel_types(mapping)
    return new_raw



def apply_notch_filter(
        raw: mne.io.Raw,
        freq: float,
        include_eog: bool = True,
        inplace: bool = False,
        suppress_warnings: bool = False,
) -> mne.io.Raw:
    """ Applies a notch filter to the given raw data. """
    assert freq > 0, f"Notch frequency must be positive"
    nyquist = raw.info['sfreq'] / 2
    if not suppress_warnings and freq >= nyquist:
        warnings.warn(
            f"Notch filter frequency ({freq:1f}Hz) " +
            f"should be less than the Nyquist frequency ({nyquist:1f}Hz). " +
            "No filter applied.",
            UserWarning
        )
        return raw
    freqs = np.arange(freq, nyquist, freq).tolist()
    assert len(freqs) > 0   # ensures at least one frequency is under Nyquist
    channel_types = ["eeg", "eog"] if include_eog else ["eeg"]
    new_raw = raw if inplace else raw.copy()
    new_raw.notch_filter(freqs=freqs, picks=channel_types, verbose=False)
    return new_raw


def apply_highpass_filter(
        raw: mne.io.Raw,
        min_freq: float,
        include_eog: bool = True,
        inplace: bool = False,
        suppress_warnings: bool = False,
) -> mne.io.Raw:
    nyquist = raw.info['sfreq'] / 2
    assert 0 < min_freq < nyquist, f"Minimum frequency must be positive and less than the Nyquist frequency ({nyquist}Hz)"
    if not suppress_warnings and min_freq > _MIN_FREQ_WARN_THRESHOLD:
        warnings.warn(
            f"High-pass filter of {min_freq}Hz is unusually high. " +
            f"Consider setting the cutoff below {_MIN_FREQ_WARN_THRESHOLD}Hz.",
            UserWarning
        )
    new_raw = raw if inplace else raw.copy()
    channel_types = ["eeg", "eog"] if include_eog else ["eeg"]
    new_raw.filter(l_freq=min_freq, h_freq=None, picks=channel_types, verbose=False)
    return new_raw


def apply_lowpass_filter(
        raw: mne.io.Raw,
        max_freq: float,
        include_eog: bool = True,
        inplace: bool = False,
        suppress_warnings: bool = False,
) -> mne.io.Raw:
    nyquist = raw.info['sfreq'] / 2
    assert 0 < max_freq < nyquist, f"Maximum frequency must be positive and less than the Nyquist frequency ({nyquist}Hz)"
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
    new_raw.filter(l_freq=None, h_freq=max_freq, picks=channel_types, verbose=False)
    return new_raw
