from typing import Optional, Union, Dict, Set, Tuple

import numpy as np
import mne

import mne_scripts.helpers.annotation_helpers as ah
import mne_scripts.helpers.event_helpers as eh
import mne_scripts.helpers.raw_helpers as rh

_TRIAL_EPOCH_BEFORE_SEC, _TRIAL_EPOCH_AFTER_SEC = 0.5, 1

_EOG_BLINK_THRESHOLD = 400e-6
_BLINK_EPOCH_BEFORE_SEC, _BLINK_EPOCH_AFTER_SEC = 0.5, 1

_ICA_NUM_COMPONENTS, _ICA_RANDOM_STATE, _ICA_MAX_ITERS = 20, 42, 800
_ICA_METHOD = 'infomax'     # 'fastica', 'picard', 'infomax'        # TODO: check if 'picard' is better


def _prepare_data(raw: mne.io.Raw, trial_events: Dict[str, int], **kwargs) -> mne.io.Raw:
    """
    Prepare the raw data for ICA by concatenating data from trial and blink epochs, optionally repeating the latter.
    The trial epochs are extracted based on the provided event codes, while the blink epochs are detected from
    eye-tracking and EOG channels. EEG data is concatenated into a single Raw object, which can be used for training ICA.
    See also `__blink_epochs` for details on blink epoch extraction.

    :param raw: MNE Raw object containing the EEG data.
    :param trial_events: Dictionary of event labels and their corresponding event codes.

    __General Keywords__
    :keyword min_freq: Minimum frequency for high-pass filtering the data. Default is None. Using a high-pass filter
        can help remove slow drifts and improve ICA performance.

    __Trial Epoch Keywords__
    :keyword trial_epoch_sec_before: Number of seconds before each trial onset to include in the epoch. Default is 0.5s.
    :keyword trial_epoch_sec_after: Number of seconds after each trial onset to include in the epoch. Default is 1s.
    :keyword trial_epoch_baseline_sec: Tuple of (start, end) times for baseline correction relative to each trial onset.
        if not specified, defaults to (-1 * trial_epoch_sec_before, 0).
    :keyword trial_reject_criteria: Dictionary of rejection criteria for trial epochs. Default is None.
    :keyword trial_reject_criteria_tmin: Minimum time for rejection criteria. Default is None.
    :keyword trial_reject_criteria_tmax: Maximum time for rejection criteria. Default is None.

    __Blink Epoch Keywords__
    :keyword et_channel: Name of the eye-tracking channel in the raw data, or None to skip eye-tracking blink detection.
    :keyword et_blink_codes: Single or set of event codes that indicate a blink in the eye-tracking channel, or None to skip.
    :keyword eog_blink_threshold: Threshold for detecting EOG blinks, 'auto' to use MNE's default threshold, or None to skip.
    :keyword blink_epoch_sec_before: Number of seconds before each blink onset to include in the epoch. Default is 0.5s.
    :keyword blink_epoch_sec_after: Number of seconds after each blink onset to include in the epoch. Default is 1s.
    :keyword blink_epoch_baseline_sec: Tuple of (start, end) times for baseline correction relative to each blink onset.
        if not specified, defaults to (-1 * blink_epoch_sec_before, 0).
    :keyword blink_epoch_repeats: Number of times to repeat the blink epochs in the final ICA data. Default is 1.

    :returns: A new MNE Raw object containing concatenated EEG data from trial and blink epochs.
    """
    # high-pass filter the data if necessary
    min_freq = kwargs.get("min_freq", None)
    if min_freq is not None:
        raw = rh.apply_highpass_filter(raw, min_freq, include_eog=False, inplace=False)

    # extract trial epochs
    all_events = eh.extract_events(raw, channel='all')
    trial_epochs_sec_before = kwargs.get("trial_epoch_sec_before", _TRIAL_EPOCH_BEFORE_SEC)
    trial_epochs = mne.Epochs(
        raw,
        events=all_events, event_id=trial_events,
        tmin=-1 * trial_epochs_sec_before,
        tmax=kwargs.get("trial_epoch_sec_after", _TRIAL_EPOCH_AFTER_SEC),
        baseline=kwargs.get("trial_epoch_baseline_sec", (-1 * trial_epochs_sec_before, 0)),
        reject=kwargs.get("trial_reject_criteria", None),
        reject_tmin=kwargs.get("trial_reject_criteria_tmin", None),
        reject_tmax=kwargs.get("trial_reject_criteria_tmax", None),
        preload=True, reject_by_annotation=True, picks="eeg"
    )
    # extract blink epochs
    blink_epoch_sec_before = kwargs.get("blink_epoch_sec_before", _BLINK_EPOCH_BEFORE_SEC)
    blink_epochs = __blink_epochs(
        raw,
        et_channel=kwargs.get("et_channel", None),
        et_blink_codes=kwargs.get("et_blink_codes", None),
        eog_blink_threshold=kwargs.get("eog_blink_threshold", _EOG_BLINK_THRESHOLD),
        before_sec=kwargs.get("blink_epoch_sec_before", _BLINK_EPOCH_BEFORE_SEC),
        after_sec=kwargs.get("blink_epoch_sec_after", _BLINK_EPOCH_AFTER_SEC),
        baseline_sec=kwargs.get("blink_epoch_baseline_sec", (-1 * blink_epoch_sec_before, 0)),
    )
    # create a unified Raw object for ICA
    trial_data = np.hstack(trial_epochs.get_data(verbose=False).copy())     # flatten epochs into 2D array
    trial_raw = mne.io.RawArray(trial_data, trial_epochs.info, verbose=False)
    blink_data = np.hstack(blink_epochs.get_data(verbose=False).copy())     # flatten epochs into 2D array
    blink_data = np.hstack([blink_data.copy() for _ in range(kwargs.get("blink_epoch_repeats", 1))])
    blink_raw = mne.io.RawArray(blink_data, blink_epochs.info, verbose=False)
    combined_raw = mne.concatenate_raws([trial_raw, blink_raw], verbose=False)
    return combined_raw


def _fit_ica(raw_for_ica: mne.io.Raw, **kwargs) -> mne.preprocessing.ICA:
    ica = mne.preprocessing.ICA(
        n_components=kwargs.get("n_components", _ICA_NUM_COMPONENTS),
        random_state=kwargs.get("random_state", _ICA_RANDOM_STATE),
        max_iter=kwargs.get("max_iter", _ICA_MAX_ITERS),
        method=kwargs.get("method", _ICA_METHOD).strip().lower(),
        fit_params=kwargs.get("fit_params", dict(extended=True))
    )
    ica.fit(
        raw_for_ica,
        reject=kwargs.get("ica_reject_criteria", dict(eeg=400e-6)),
        reject_by_annotation=True, picks=["eeg", "eog"],
    )
    if kwargs.get("visualize_components", True):
        # inspect components to make sure ICA worked as expected
        ica.plot_components(picks=range(20), block=True)
    return ica


def __blink_epochs(
        raw: mne.io.Raw,
        et_channel: Optional[str] = None,
        et_blink_codes: Optional[Union[int, Set[int]]] = None,
        eog_blink_threshold: Optional[Union[float, str]] = _EOG_BLINK_THRESHOLD,
        before_sec: float = _BLINK_EPOCH_BEFORE_SEC,
        after_sec: float = _BLINK_EPOCH_AFTER_SEC,
        baseline_sec: Tuple[float, float] = (-1 * _BLINK_EPOCH_BEFORE_SEC, 0)
) -> mne.Epochs:
    """
    Detect blink onsets from eye-tracking and EOG channels and create epochs around each blink onset.
    By default, epochs are based on EOG detected blinks, using a threshold of 400μV. To prevent EOG blink detection, set
    the `eog_blink_threshold` parameter to None.
    To detect blinks from eye-tracking, provide the `et_channel` and `et_blink_codes` parameters. If both arguments are
    None, no eye-tracking blinks will be detected. If one is None and the other is not, a ValueError will be raised.

    :param raw: MNE Raw object containing the EEG data.
    :param et_channel: Name of the eye-tracking channel in the raw data, or None to skip eye-tracking blink detection.
    :param et_blink_codes: Single or set of event codes that indicate a blink in the eye-tracking channel, or None to skip.
    :param eog_blink_threshold: Threshold for detecting EOG blinks, 'auto' to use MNE's default threshold, or None to skip.
    :param before_sec: Number of seconds before each blink onset to include in the epoch.
    :param after_sec: Number of seconds after each blink onset to include in the epoch.
    :param baseline_sec: Tuple of (start, end) times for baseline correction relative to each blink onset.
    """
    assert before_sec >= 0, "before_ms must be non-negative"
    assert after_sec >= 0, "after_ms must be non-negative"
    new_raw = raw.copy()        # copy to avoid modifying the original raw data
    # clear existing annotations to prevent blink detection from ignoring subset of the data:
    new_raw.set_annotations(mne.Annotations([], [], []))
    # detect blinks from eyetracking and EOG
    new_raw.set_annotations(ah.blink_annotations(
        new_raw, et_channel, et_blink_codes, eog_blink_threshold,
        0, 1, 1     # we only need blink onsets to create epochs
    ))
    blink_epochs = mne.Epochs(
        new_raw,
        tmin=-1 * before_sec, tmax=after_sec, baseline=baseline_sec,
        events=None, preload=True, reject_by_annotation=False, reject=None,     # use annotations for epoching
        picks='eeg', verbose=False
    )
    return blink_epochs
