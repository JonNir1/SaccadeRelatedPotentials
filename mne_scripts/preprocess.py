import os
import time

import mne

import mne_scripts.helpers.raw_helpers as rawh
import mne_scripts.helpers.annotation_helpers as annh

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
    :keyword examine_psd: bool. Whether to inspect the PSD of the data. Default is True.
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
        step1_raw = rawh.set_montage(raw, montage=kwargs.get("montage", None), overwrite=True)
        step1_raw = rawh.remap_channels(
            step1_raw,
            eog_channels=kwargs.get("eog_channels", None),
            stim_channels=kwargs.get("stim_channels", None),
            gaze_channels=kwargs.get("gaze_channels", None),
            pupil_channels=kwargs.get("pupil_channels", None),
            misc_channels=kwargs.get("misc_channels", None),
        )
        # resample the data
        if "resample_freq" in kwargs.keys():
            step1_raw = rawh.resample(step1_raw, kwargs.get("resample_freq"), inplace=True)
        # inspect for noisy/silent channels
        step1_raw.plot(
            scalings=kwargs.get("scalings", _VISUALIZATION_SCALING), block=kwargs.get("block", True),
            title="Mark very noisy/completely silent channels"
        )
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
        step2_raw = rawh.apply_highpass_filter(
            step2_raw,
            min_freq=kwargs.get("min_freq", _MIN_FREQ),
            include_eog=True,
            inplace=True,
            suppress_warnings=False,
        )
        step2_raw = rawh.apply_lowpass_filter(
            step2_raw,
            max_freq=kwargs.get("max_freq", _MAX_FREQ),
            include_eog=True,
            inplace=True,
            suppress_warnings=False,
        )
        step2_raw = rawh.apply_notch_filter(
            step2_raw,
            freq=kwargs.get("notch_freq", _NOTCH_FREQ),
            include_eog=True,
            inplace=True,
        )
        step2_raw = rawh.set_reference(
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
    :keyword examine_psd: bool. Whether to inspect the PSD of the data. Default is True.
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
        jump_annotations = annh.voltage_jump_annotations(
            step3_raw,
            channel_type='eeg',
            jump_threshold_volts=kwargs.get("jump_threshold_volts", _VOLTAGE_JUMP_THRESHOLD),
            jump_window_ms=kwargs.get("jump_window_ms", _VOLTAGE_JUMP_DURATION_MS),
            min_channel_ratio=kwargs.get("min_channel_ratio", _VOLTAGE_JUMP_MIN_RATIO),
            pre_annotation_ms=kwargs.get("pre_annotation_ms", _BAD_VOLTAGE_PRE_MS),
            post_annotation_ms=kwargs.get("post_annotation_ms", _BAD_VOLTAGE_POST_MS),
            merge_within_ms=kwargs.get("merge_within_ms", _BAD_VOLTAGE_MERGE_MS)
        )
        step3_raw.set_annotations(jump_annotations)
        step3_raw.plot(     # verify periods of bad data
            n_channels=20, scalings=kwargs.get("scalings", _VISUALIZATION_SCALING), block=kwargs.get("block", True),
            title="Verify periods of bad data"
        )
        # inspect PSD
        if kwargs.get("examine_psd", True):
            # check if all channels are "bundled" together and reject channels that are outside the norm
            spectrum = step3_raw.copy().compute_psd(
                picks=['eeg'],
                n_fft=kwargs.get("psd_nfft", _PSD_FFT_COMPONENTS),
                reject_by_annotation=True,  # exclude bad periods
                exclude='bads',             # exclude bad channels
                verbose=False,
            )
            spectrum.plot(picks=['eeg', 'eog'], exclude='bads',)
            step3_raw.plot(                 # mark bad channels based on PSD inspection
                n_channels=20, scalings=kwargs.get("scalings", _VISUALIZATION_SCALING), block=kwargs.get("block", True),
                title="Mark bad channels based on PSD inspection"
            )
        # interpolate bad channels
        if kwargs.get("interpolate_bads", True):
            step3_raw.interpolate_bads(reset_bads=True)
        step3_raw.save(save_to, overwrite=True)
        t1 = time.time()
        if verbose:
            elapsed = t1 - t0
            print(f"\tStep 3 completed ({elapsed:.2f} sec).")
        return step3_raw
