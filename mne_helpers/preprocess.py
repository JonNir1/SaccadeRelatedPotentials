import os
import time
import warnings
from typing import Optional, List

import numpy as np
import mne

import mne_helpers.utils as u

_MIN_FREQ, _MAX_FREQ, _NOTCH_FREQ = 0.1, 100, 50
_REF_ELECTRODE = "average"
_VOLTAGE_JUMP_THRESHOLD, _VOLTAGE_JUMP_DURATION_MS, _VOLTAGE_JUMP_MIN_RATIO = 2e-4, 100, 0.5
_BAD_VOLTAGE_PRE_MS, _BAD_VOLTAGE_POST_MS, _BAD_VOLTAGE_MERGE_MS = 250, 250, 50
_PSD_FFT_COMPONENTS = 1024
_VISUALIZATION_SCALING = dict(eeg=1e-4, eog=1e-4, eyegaze=5e2, pupil=5e2)


def preprocess_raw_fif(path: str, **kwargs) -> mne.io.Raw:
    """
    Loads an EEG `.fif` file and performs interactive and automated preprocessing on it, if not already preprocessed.
    The preprocessing pipeline includes the following steps (see keyword arguments for customization):
    Step 1:
        1. Set the montage for the data.
        2. Remap channels to new types based on user-provided mappings.
        3. Resample the data to a new frequency, if provided.
        4. Manually inspect the data for noisy or silent channels, interpolating if requested.
        5. Save the preprocessed data to a file for future use.
    Step 2:
        1. Apply high-pass filtering to the data.
        2. Apply low-pass filtering to the data.
        3. Apply notch filtering to remove line noise.
        4. Set the reference for the data.
        5. Save the preprocessed data to a file for future use.
    Step 3:
        1. Detect noisy voltage jumps in the data.
        2. Interpolate bad channels if requested.
        3. Save the preprocessed data to a file for future use.

    :param path: The file path to the EEG data file.

    __ Step 1 Keyword Arguments __
    :keyword montage: str or mne.channels.Montage. Montage to set for the data.
    :keyword eog_channels: list of str. Channel names to be marked as EOG.
    :keyword stim_channels: list of str. Channel names to be marked as stimulus channels.
    :keyword gaze_channels: list of str. Channel names to be marked as eye gaze channels.
    :keyword pupil_channels: list of str. Channel names to be marked as pupil channels.
    :keyword misc_channels: list of str. Channel names to be marked as miscellaneous channels.
    :keyword resample_freq: float. If provided, resample the data to this frequency (Hz).

    __ Step 2 Keyword Arguments __
    :keyword min_freq: float. Low-frequency cutoff for EEG & EOG high-pass filtering (Hz). Default is 0.1 Hz.
    :keyword max_freq: float. High-frequency cutoff for EEG & EOG low-pass filtering (Hz). Default is 100 Hz.
    :keyword notch_freq: float. Frequency for notch filtering to remove line noise. Default is 50 Hz.
    :keyword ref_channel: str. Reference channel for the data. Default is "average".
    :keyword ref_eog: bool. Whether to include EOG channels in the reference. Default is True.

    __ Step 3 Keyword Arguments __
    :keyword jump_threshold_volts: float. Voltage threshold (in volts) to detect jumps. Default is 2e-4 (200 μV).
    :keyword jump_window_ms: int. Window size (in ms) for computing voltage jumps. Default is 100 ms.
    :keyword min_channel_ratio: float. Minimum ratio of EEG channels exceeding the threshold to flag a jump. Default is 0.5.
    :keyword pre_annotation_ms: float. Time (in ms) to annotate before a detected jump. Default is 250 ms.
    :keyword post_annotation_ms: float. Time (in ms) to annotate after a detected jump. Default is 250 ms.
    :keyword merge_within_ms: float. Merge annotations of the same kind if closer than this time (ms). Default is 50 ms.
    :keyword psd_nfft: int. Number of FFT points for PSD estimation. Default is 1024.

    __ General Keyword Arguments __
    :keyword scalings: dict. MNE scaling for visualization.
    :keyword block: bool. Whether plots block execution for manual inspection (should be True).
    :keyword interpolate_bads: bool. Whether to interpolate bad channels. Default is True.
    :keyword copy_metadata: bool. Whether to copy metadata from the preprocessed data to the raw data. Default is True.
    :keyword verbose: bool. Whether to print verbose output (default is True).

    :returns: A preprocessed copy of the raw MNE object with annotations, filtering, and re-referencing applied.
    """
    basedir, filename = os.path.dirname(path), os.path.basename(path).split('.')[0]
    raw = mne.io.Raw(path)
    step1_path = os.path.join(basedir, f"{filename}-step1.fif")
    step1_raw = _step_1(raw, step1_path, **kwargs)
    step2_path = os.path.join(basedir, f"{filename}-step2.fif")
    step2_raw = _step_2(step1_raw, step2_path, **kwargs)
    step3_raw = _step_3(step2_raw, os.path.join(basedir, f"{filename}-step3.fif"), **kwargs)

    if kwargs.get("copy_metadata", True):
        raw.info = step3_raw.info
        raw.set_annotations(step3_raw.annotations)
        raw.save(path, overwrite=True)

        step1_raw.info = step3_raw.info
        step1_raw.set_annotations(step3_raw.annotations)
        step1_raw.save(step1_path, overwrite=True)

        step2_raw.info = step3_raw.info
        step2_raw.set_annotations(step3_raw.annotations)
        step2_raw.save(step2_path, overwrite=True)

    return step3_raw



def _step_1(raw: mne.io.Raw, save_to: str, **kwargs) -> mne.io.Raw:
    """
    Loads a step1-preprocessed data file if it exists, otherwise performs the first steps of preprocessing:
        1. Set the montage for the data.
        2. Remap channels to new types based on user-provided mappings.
        3. Resample the data to a new frequency, if provided.
        4. Manually inspect the data for noisy or silent channels, interpolating if requested.
        5. Save the preprocessed data to a file for future use.

    :param raw: The raw EEG data to preprocess (MNE Raw object).
    :param save_to: The file path to save the preprocessed data to.

    :keyword montage: str or mne.channels.Montage. Montage to set for the data.
    :keyword eog_channels: list of str. Channel names to be marked as EOG.
    :keyword stim_channels: list of str. Channel names to be marked as stimulus channels.
    :keyword gaze_channels: list of str. Channel names to be marked as eye gaze channels.
    :keyword pupil_channels: list of str. Channel names to be marked as pupil channels.
    :keyword misc_channels: list of str. Channel names to be marked as miscellaneous channels.
    :keyword resample_freq: float. If provided, resample the data to this frequency (Hz).
    :keyword scalings: dict. MNE scaling for visualization.
    :keyword block: bool. Whether plots block execution for manual inspection (should be True).
    :keyword interpolate_bads: bool. Whether to interpolate bad channels.
    :keyword verbose: bool. Whether to print verbose output (default is True).

    :returns: A step1-preprocessed copy of the mne.io.Raw object, with the first steps of preprocessing applied.
    """
    os.makedirs(os.path.dirname(save_to), exist_ok=True)
    try:
        step1_raw = mne.io.read_raw_fif(save_to, preload=True)
        return step1_raw
    except FileNotFoundError:
        t0 = time.time()
        verbose = kwargs.get("verbose", True)
        if verbose:
            print("Preprocessing Step 1: Setting montage, remapping channels, and resampling data...")
        step1_raw = __set_montage(raw, montage=kwargs.get("montage", None), overwrite=True)
        step1_raw = __remap_channels(
            step1_raw,
            eog_channels=kwargs.get("eog_channels", None),
            stim_channels=kwargs.get("stim_channels", None),
            gaze_channels=kwargs.get("gaze_channels", None),
            pupil_channels=kwargs.get("pupil_channels", None),
            misc_channels=kwargs.get("misc_channels", None),
        )
        # resample the data
        if "resample_freq" in kwargs.keys():
            step1_raw = u.resample(step1_raw, kwargs.get("resample_freq"), inplace=True)
        # inspect for noisy/silent channels
        step1_raw.plot(scalings=kwargs.get("scalings", _VISUALIZATION_SCALING), block=kwargs.get("block", True))
        if kwargs.get("interpolate_bads", True):
            step1_raw.interpolate_bads(reset_bads=True)
        step1_raw.save(save_to, overwrite=True)
        t1 = time.time()
        if verbose:
            elapsed = t1 - t0
            print(f"\tStep 1 completed ({elapsed:.2f} sec).")
        return step1_raw


def _step_2(raw: mne.io.Raw, save_to: str, **kwargs) -> mne.io.Raw:
    """
    Loads a step2-preprocessed data file if it exists, otherwise performs the second steps of preprocessing:
        1. Apply high-pass filtering to the data.
        2. Apply low-pass filtering to the data.
        3. Apply notch filtering to remove line noise.
        4. Set the reference for the data.
        5. Save the preprocessed data to a file for future use.

    :param raw: The raw EEG data to preprocess (MNE Raw object).
    :param save_to: The file path to save the preprocessed data to.

    :keyword min_freq: float. Low-frequency cutoff for EEG & EOG high-pass filtering (Hz). Default is 0.1 Hz.
    :keyword max_freq: float. High-frequency cutoff for EEG & EOG low-pass filtering (Hz). Default is 100 Hz.
    :keyword notch_freq: float. Frequency for notch filtering to remove line noise. Default is 50 Hz.
    :keyword ref_channel: str. Reference channel for the data. Default is "average".
    :keyword ref_eog: bool. Whether to include EOG channels in the reference. Default is True.
    :keyword verbose: bool. Whether to print verbose output (default is True).

    :returns: A step2-preprocessed copy of the mne.io.Raw object, with the second steps of preprocessing applied.
    """
    os.makedirs(os.path.dirname(save_to), exist_ok=True)
    try:
        step2_raw = mne.io.read_raw_fif(save_to, preload=True)
        return step2_raw
    except FileNotFoundError:
        t0 = time.time()
        verbose = kwargs.get("verbose", True)
        if verbose:
            print("Preprocessing Step 2: filtering and re-referencing the data...")
        step2_raw = raw.copy()
        step2_raw = u.apply_highpass_filter(
            step2_raw,
            min_freq=kwargs.get("min_freq", _MIN_FREQ),
            include_eog=True,
            inplace=True,
            suppress_warnings=False,
        )
        step2_raw = u.apply_lowpass_filter(
            step2_raw,
            max_freq=kwargs.get("max_freq", _MAX_FREQ),
            include_eog=True,
            inplace=True,
            suppress_warnings=False,
        )
        step2_raw = u.apply_notch_filter(
            step2_raw,
            freq=kwargs.get("notch_freq", _NOTCH_FREQ),
            include_eog=True,
            inplace=True,
        )
        step2_raw = u.set_reference(
            step2_raw,
            ref_channel=kwargs.get("ref_channel", _REF_ELECTRODE),
            include_eog=kwargs.get("ref_eog", True),
        )
        step2_raw.save(save_to, overwrite=True)
        t1 = time.time()
        if verbose:
            elapsed = t1 - t0
            print(f"\tStep 2 completed ({elapsed:.2f} sec).")
        return step2_raw


def _step_3(raw: mne.io.Raw, save_to: str, **kwargs) -> mne.io.Raw:
    """
    Loads a step3-preprocessed data file if it exists, otherwise performs the third steps of preprocessing:
        1. Detect noisy voltage jumps in the data.
        2. Interpolate bad channels if requested.
        3. Save the preprocessed data to a file for future use.

    :param raw: The raw EEG data to preprocess (MNE Raw object).
    :param save_to: The file path to save the preprocessed data to.

    :keyword jump_threshold_volts: float. Voltage threshold (in volts) to detect jumps. Default is 2e-4 (200 μV).
    :keyword jump_window_ms: int. Window size (in ms) for computing voltage jumps. Default is 100 ms.
    :keyword min_channel_ratio: float. Minimum ratio of EEG channels exceeding the threshold to flag a jump. Default is 0.5.
    :keyword pre_annotation_ms: float. Time (in ms) to annotate before a detected jump. Default is 250 ms.
    :keyword post_annotation_ms: float. Time (in ms) to annotate after a detected jump. Default is 250 ms.
    :keyword merge_within_ms: float. Merge annotations of the same kind if closer than this time (ms). Default is 50 ms.
    :keyword psd_nfft: int. Number of FFT points for PSD estimation. Default is 1024.
    :keyword scalings: dict. MNE scaling for visualization.
    :keyword block: bool. Whether plots block execution for manual inspection (should be True).
    :keyword interpolate_bads: bool. Whether to interpolate bad channels. Default is True.
    :keyword verbose: bool. Whether to print verbose output (default is True).

    :returns: A step3-preprocessed copy of the mne.io.Raw object, with the third steps of preprocessing applied.
    """
    os.makedirs(os.path.dirname(save_to), exist_ok=True)
    try:
        step3_raw = mne.io.read_raw_fif(save_to, preload=True)
        return step3_raw
    except FileNotFoundError:
        t0 = time.time()
        verbose = kwargs.get("verbose", True)
        if verbose:
            print("Preprocessing Step 3: detecting voltage jumps and inspecting PSD...")
        step3_raw = raw.copy()
        # detect noisy voltage jumps
        jump_annotations = __annotate_voltage_jumps(
            step3_raw, channel_type='eeg',
            jump_threshold_volts=kwargs.get("jump_threshold_volts", _VOLTAGE_JUMP_THRESHOLD),
            jump_window_ms=kwargs.get("jump_window_ms", _VOLTAGE_JUMP_DURATION_MS),
            min_channel_ratio=kwargs.get("min_channel_ratio", _VOLTAGE_JUMP_MIN_RATIO),
            pre_annotation_ms=kwargs.get("pre_annotation_ms", _BAD_VOLTAGE_PRE_MS),
            post_annotation_ms=kwargs.get("post_annotation_ms", _BAD_VOLTAGE_POST_MS),
            merge_within_ms=kwargs.get("merge_within_ms", _BAD_VOLTAGE_MERGE_MS)
        )
        step3_raw.set_annotations(jump_annotations)
        step3_raw.plot(     # verify periods of bad data
            n_channels=20, scalings=kwargs.get("scalings", _VISUALIZATION_SCALING), block=kwargs.get("block", True)
        )
        # inspect PSD
        spectrum = step3_raw.copy().compute_psd(
            picks=['eeg'],
            n_fft=kwargs.get("psd_nfft", _PSD_FFT_COMPONENTS),
            reject_by_annotation=True,      # exclude bad periods
            exclude='bads',                 # exclude bad channels
            verbose=False,
        )
        spectrum.plot(
            # check if all channels are "bundled" together and reject if a channel is too high/low
            picks=['eeg'], exclude='bads', block=kwargs.get("block", True)
        )
        if kwargs.get("interpolate_bads", True):
            step3_raw.interpolate_bads(reset_bads=True)
        step3_raw.save(save_to, overwrite=True)
        t1 = time.time()
        if verbose:
            elapsed = t1 - t0
            print(f"\tStep 3 completed ({elapsed:.2f} sec).")
        return step3_raw


def __set_montage(raw: mne.io.Raw, montage: Optional[str] = None, overwrite: bool = False) -> mne.io.Raw:
    """ Set the montage for the data, optionally overwriting the existing montage. """
    new_raw = raw.copy()
    if new_raw.get_montage() is None:
        new_raw.set_montage(montage, on_missing='ignore', verbose=False)
        return new_raw
    warnings.warn(f"Attempting to set montage '{montage}' on data with existing montage.")
    if not overwrite:
        return new_raw
    new_raw.set_montage(montage, on_missing='ignore', verbose=False)
    return new_raw


def __remap_channels(
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


def __annotate_voltage_jumps(
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
    return u.merge_annotations(annotations, merge_within_ms)
