import numpy as np
import mne

import mne_helpers.utils as u


def preprocess(raw: mne.io.Raw, **kwargs) -> mne.io.Raw:
    """
    Perform interactive and automated preprocessing on a Raw MNE object. This includes
    manual inspection, resampling, filtering, artifact detection, and re-referencing.

    The preprocessing pipeline performs the following steps:
        1. Visual inspection for noisy or silent channels.
        2. Optional resampling of the data.
        3. Marking EOG channels for proper filtering.
        4. Band-pass and notch filtering.
        5. Automated detection and annotation of voltage jumps.
        6. Power spectral density (PSD) visualization for channel quality.
        7. Re-referencing of EEG data.
        8. Final manual inspection (optional).

    :param raw: The raw EEG data to preprocess (MNE Raw object).

    :keyword new_freq: Optional float. If provided, resample the data to this frequency (Hz).
    :keyword eog_channels: Optional list of str. Channel names to be marked as EOG.
    :keyword max_freq: Optional float. High-frequency cutoff for EEG filtering (Hz). Defaults to Nyquist (0.5 × sfreq).
    :keyword min_freq: Optional float. Low-frequency cutoff for EEG/EOG filtering (Hz). Default is 0.1 Hz.
    :keyword notch_freq: Optional float. Frequency for notch filtering to remove line noise. Default is 50 Hz.
    :keyword jump_threshold_volts: Optional float. Voltage threshold (in volts) to detect jumps. Default is 2e-4 (200 μV).
    :keyword jump_window_ms: Optional int. Window size (in ms) for computing voltage jumps. Default is 100 ms.
    :keyword min_channel_ratio: Optional float. Minimum ratio of EEG channels exceeding the threshold to flag a jump. Default is 0.5.
    :keyword pre_annotation_ms: Optional float. Time (in ms) to annotate before a detected jump. Default is 250 ms.
    :keyword post_annotation_ms: Optional float. Time (in ms) to annotate after a detected jump. Default is 250 ms.
    :keyword merge_within_ms: Optional float. Merge annotations of the same kind if closer than this time (ms). Default is 50 ms.
    :keyword psd_nfft: Optional int. Number of FFT points for PSD estimation. Default is 1024.
    :keyword ref_channel: Optional str or list of str. EEG re-referencing strategy. Default is "average".
    :keyword visualization_scales: Optional dict. Scaling for visualization. Default is suitable for EEG, EOG, and eye-tracking.
    :keyword visualization_blocks: Optional bool. Whether plots block execution for manual inspection. Default is True.

    :returns: A preprocessed copy of the raw MNE object with annotations, filtering, and re-referencing applied.
    """
    new_raw = raw.copy()

    # step 1: inspect for always silent / always noisy channels
    visualization_scales = kwargs.get("visualization_scales", dict(eeg=1e-4, eog=1e-4, eyegaze=5e2, pupil=5e2))
    visualization_blocks = kwargs.get("visualization_blocks", True)
    ## find silent/noisy channels:
    new_raw.plot(n_channels=20, scalings=visualization_scales, block=visualization_blocks)

    # step 2: resample the data
    new_freq = kwargs.get("new_freq", None)
    if new_freq is not None and new_freq > 0:
        events = u.extract_events(new_raw, channel='all')
        new_raw, new_events = new_raw.resample(new_freq, events=events)

    # step 3: mark EOG channels     # TODO: check if this should be done later?
    new_raw.set_channel_types({ch: 'eog' for ch in kwargs.get("eog_channels", [])})

    # step 4: filter the data
    new_raw = _filter_raw(
        new_raw,
        max_freq=kwargs.get("max_freq", 0.5 * raw.info['sfreq']),       # default to Nyquist frequency
        min_freq=kwargs.get("min_freq", 0.1),                           # default to 0.1 Hz
        notch_freq=kwargs.get("notch_freq", 50),                        # default to 50 Hz
    )

    # step 5: detect noisy voltage jumps
    jump_annotations = _annotate_voltage_jumps(
        new_raw, channel_type='eeg',
        jump_threshold_volts=kwargs.get("jump_threshold_volts", 2e-4),  # default to 200 μV
        jump_window_ms=kwargs.get("jump_window_ms", 100),               # default to 100 ms
        min_channel_ratio=kwargs.get("min_channel_ratio", 0.5),         # default to 50%
        pre_annotation_ms=kwargs.get("pre_annotation_ms", 250),         # default to 250 ms
        post_annotation_ms=kwargs.get("post_annotation_ms", 250),       # default to 250 ms
        merge_within_ms=kwargs.get("merge_within_ms", 50)               # default to 50 ms
    )
    new_raw.set_annotations(jump_annotations)
    ## check for periods of bad data:
    new_raw.plot(n_channels=20, scalings=visualization_scales, block=visualization_blocks)

    # step 6: inspect PSD
    ## check if all channels are "bundled" together and reject if a channel is too high/low
    spectrum = new_raw.compute_psd(
        picks=['eeg'],
        n_fft=kwargs.get("psd_nfft", 1024),                             # defaults to 1024
        reject_by_annotation=True,
    )
    spectrum.plot()

    # step 7: re-reference eeg data
    new_raw.set_eeg_reference(
        ref_channels=kwargs.get("ref_channel", "average"),              # defaults to average reference
        verbose=False
    )
    ## check for any remaining bad channels
    new_raw.plot(n_channels=20, scalings=visualization_scales, block=visualization_blocks)
    return new_raw


def _filter_raw(
        raw: mne.io.Raw, max_freq: float, min_freq: float, notch_freq: float,
) -> mne.io.Raw:
    assert 0 < min_freq < max_freq, "Unsuitable low/high frequency values"
    assert 0 < notch_freq, "Unsuitable notch frequency value"
    new_raw = raw.copy()
    # remove high-frequency noise only in EEG, to maintain muscle activity in EOG:
    new_raw.filter(h_freq=max_freq, picks=['eeg'])
    # remove low-frequency noise, including in EOG:
    new_raw.filter(l_freq=min_freq, picks=['eeg', 'eog'])
    # remove AC line noise, including from EOG:
    new_raw.notch_filter(
        freqs=np.arange(notch_freq, 1 + 6 * notch_freq, notch_freq), picks=['eeg', 'eog'],
    )
    return new_raw


def _annotate_voltage_jumps(
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
