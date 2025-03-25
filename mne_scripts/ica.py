from typing import Optional, Union, Dict, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import mne

import mne_scripts.helpers.annotation_helpers as ah
import mne_scripts.helpers.event_helpers as eh
import mne_scripts.helpers.raw_helpers as rh

_TRIAL_EPOCHS_BEFORE_SEC, _TRIAL_EPOCH_AFTER_SEC = 0.5, 1

_EOG_BLINK_THRESHOLD = 400e-6
_BLINK_EPOCH_BEFORE_SEC, _BLINK_EPOCH_AFTER_SEC = 0.5, 1

_ICA_NUM_COMPONENTS, _ICA_RANDOM_STATE, _ICA_MAX_ITERS = 20, 42, 800
_ICA_METHOD = 'infomax'     # 'fastica', 'picard', 'infomax'        # TODO: check if 'picard' is better
_ICA_PSD_FMAX = 60


def run_ica(raw: mne.io.Raw, trial_events: Dict[str, int], **kwargs):
    """
    __General Keywords__
    :keyword min_freq: Minimum frequency for high-pass filtering the data. Default is None. Using a high-pass filter
        can help remove slow drifts and improve ICA performance.
    :keyword epoch_with_eog: Whether to include EOG channels in the epoch data. Default is False.
    """
    return None


def _prepare_data(raw: mne.io.Raw, trial_events: Dict[str, int], **kwargs) -> (mne.io.Raw, mne.Epochs):
    """
    Prepare the raw data for ICA by concatenating data from trial and blink epochs, optionally repeating the latter.
    The trial epochs are extracted based on the provided event codes, while the blink epochs are detected from
    eye-tracking and EOG channels. EEG data is concatenated into a single Raw object, which can be used for training ICA.
    See also `__extract_trial_epochs` and `__extract_blink_epochs` for details on epoch extraction.

    :param raw: MNE Raw object containing the EEG data.
    :param trial_events: Dictionary of event labels and their corresponding event codes.

    __General Keywords__
    :keyword min_freq: Minimum frequency for high-pass filtering the data. Default is None. Using a high-pass filter
        can help remove slow drifts and improve ICA performance.
    :keyword epoch_with_eog: Whether to include EOG channels in the epoch data. Default is False.

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

    :returns:
        - raw_for_ica: A new MNE Raw object containing concatenated EEG data from trial and blink epochs.
        - trial_epochs: MNE Epochs object containing the trial epochs.
    """
    # high-pass filter the data if necessary
    min_freq = kwargs.get("min_freq", None)
    if min_freq is not None:
        raw = rh.apply_highpass_filter(raw, min_freq, include_eog=False, inplace=False)

    # extract trial and blink epochs
    trial_epochs = __extract_trial_epochs(
        raw, trial_events,
        sec_before=kwargs.get("trial_epoch_sec_before", _TRIAL_EPOCHS_BEFORE_SEC),
        sec_after=kwargs.get("trial_epoch_sec_after", _TRIAL_EPOCH_AFTER_SEC),
        baseline_sec=kwargs.get("trial_epoch_baseline_sec", None),
        reject_criteria=kwargs.get("trial_reject_criteria", None),
        reject_criteria_tmin=kwargs.get("trial_reject_criteria_tmin", None),
        reject_criteria_tmax=kwargs.get("trial_reject_criteria_tmax", None),
        use_eog=kwargs.get("epoch_with_eog", False),
    )
    blink_epochs = __extract_blink_epochs(
        raw,
        sec_before=kwargs.get("blink_epoch_sec_before", _BLINK_EPOCH_BEFORE_SEC),
        sec_after=kwargs.get("blink_epoch_sec_after", _BLINK_EPOCH_AFTER_SEC),
        baseline_sec=kwargs.get("blink_epoch_baseline_sec", None),
        et_channel=kwargs.get("et_channel", None),
        et_blink_codes=kwargs.get("et_blink_codes", None),
        eog_threshold=kwargs.get("eog_blink_threshold", _EOG_BLINK_THRESHOLD),
        use_eog=kwargs.get("epoch_with_eog", False),
    )

    # create a unified Raw object for ICA
    trial_data = np.hstack(trial_epochs.get_data(verbose=False).copy())     # flatten epochs into 2D array
    trial_raw = mne.io.RawArray(trial_data, trial_epochs.info, verbose=False)
    blink_data = np.hstack(blink_epochs.get_data(verbose=False).copy())     # flatten epochs into 2D array
    blink_data = np.hstack([blink_data.copy() for _ in range(kwargs.get("blink_epoch_repeats", 1))])
    blink_raw = mne.io.RawArray(blink_data, blink_epochs.info, verbose=False)
    raw_for_ica = mne.concatenate_raws([trial_raw, blink_raw], verbose=False)
    del trial_raw, blink_raw, blink_epochs
    return raw_for_ica, trial_epochs


def _fit_ica(raw_for_ica: mne.io.Raw, **kwargs) -> mne.preprocessing.ICA:
    """
    Fit an Independent Component Analysis (ICA) model to the provided raw data. The ICA model is trained on the
    concatenated EEG data from trial and blink epochs, and can be used to identify and remove artifacts from the data.

    :param raw_for_ica: MNE Raw object to train the ICA model on.

    :keyword num_components: Number of components to extract from the data. Default is 20.
    :keyword random_state: Random seed for the ICA algorithm. Default is 42.
    :keyword max_iter: Maximum number of iterations for the ICA algorithm. Default is 800.
    :keyword method: ICA algorithm to use, one of 'fastica', 'picard', or 'infomax'. Default is 'infomax'.
    :keyword fit_params: Additional parameters to pass to the ICA fit method. Default is {'extended': True}.
    :keyword ica_reject_criteria: Dictionary of rejection criteria for ICA fitting. Default is {'eeg': 400e-6}.
    """
    num_components = kwargs.get("num_components", _ICA_NUM_COMPONENTS)
    ica = mne.preprocessing.ICA(
        n_components=num_components,
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
    return ica


def _visualize_ica(
        ica: mne.preprocessing.ICA, raw: mne.io.Raw, trial_epochs: Optional[mne.Epochs] = None, **kwargs
):
    """
    Visualize the ICA components as topographic maps and time series.
    If the `plot_single_components` keyword is True, each component is plotted individually with its PSD and time course.
    If the `plot_ica_psd` keyword is True, the PSD of the ICA sources is plotted.
    Execution is paused after all plots are rendered, and continues when the user presses Enter.

    :param ica: ICA object containing the fitted ICA model.
    :param raw: Raw object containing the EEG data.
    :param trial_epochs: Epochs object containing the trial epochs for plotting individual component properties, or None
        to skip this step.
    :keyword plot_ica_psd: Whether to plot the PSD of the ICA sources. Default is False.
    """
    num_components = ica.n_components
    ica.plot_components(picks=range(num_components))
    ica.plot_sources(raw)
    if trial_epochs is not None:
        for i in range(num_components):
            ica.plot_properties(trial_epochs, picks=i, psd_args={'fmax': _ICA_PSD_FMAX}, verbose=False)
    if kwargs.get("plot_ica_psd", False):
        ica_raw = ica.get_sources(raw)
        ica_raw.set_channel_types({name: "eeg" for name in ica_raw.ch_names})
        ica_raw.info["bads"] = []
        spectrum = ica_raw.copy().compute_psd(
            picks=['eeg'],
            n_fft=int(2 * raw.info['sfreq']),
            n_overlap=int(0.1 * raw.info['sfreq']),
            fmax=_ICA_PSD_FMAX,
            verbose=False,
        )
        spectrum.plot()
    plt.pause(0.01)
    input("Press Enter to continue...")
    return


def _apply_ica(ica: mne.preprocessing.ICA, raw: mne.io.Raw) -> mne.io.Raw:
    # TODO: Continue from here
    return None



def __extract_trial_epochs(
        raw: mne.io.Raw,
        trial_events: Dict[str, int],
        sec_before: float = _TRIAL_EPOCHS_BEFORE_SEC,
        sec_after: float = _TRIAL_EPOCH_AFTER_SEC,
        baseline_sec: Optional[Tuple[float, float]] = None,
        reject_criteria: Optional[Dict[str, float]] = None,
        reject_criteria_tmin: Optional[float] = None,
        reject_criteria_tmax: Optional[float] = None,
        use_eog: bool = True,
) -> mne.Epochs:
    """
    Extract trial epochs from the raw data based on the provided event codes. By default, epochs are extracted from EEG
    channels only, but EOG channels can be included by setting the `use_eog` parameter to True.

    :param raw: MNE Raw object containing the EEG data.
    :param trial_events: Dictionary of event labels and their corresponding event codes.
    :param sec_before: Number of seconds before each trial onset to include in the epoch.
    :param sec_after: Number of seconds after each trial onset to include in the epoch.
    :param baseline_sec: Tuple of (start, end) times for baseline correction relative to each trial onset.
    :param reject_criteria: Dictionary of rejection criteria for trial epochs.
    :param reject_criteria_tmin: Minimum time for rejection criteria.
    :param reject_criteria_tmax: Maximum time for rejection criteria.
    :param use_eog: Whether to include EOG channels in the epoch data.

    :returns: MNE Epochs object containing the trial epochs.
    """
    all_events = eh.extract_events(raw, channel='all')
    baseline_sec = baseline_sec or (-1 * sec_before, 0)
    picks = ["eeg", "eog"] if use_eog else ["eeg"]
    trial_epochs = mne.Epochs(
        raw,
        events=all_events, event_id=trial_events,
        tmin=-1 * sec_before,
        tmax=sec_after,
        baseline=baseline_sec,
        reject=reject_criteria,
        reject_tmin=reject_criteria_tmin,
        reject_tmax=reject_criteria_tmax,
        picks=picks,
        preload=True,
        reject_by_annotation=True,
    )
    return trial_epochs


def __extract_blink_epochs(
        raw: mne.io.Raw,
        sec_before: float = _BLINK_EPOCH_BEFORE_SEC,
        sec_after: float = _BLINK_EPOCH_AFTER_SEC,
        baseline_sec: Optional[Tuple[float, float]] = None,
        et_channel: Optional[str] = None,
        et_blink_codes: Optional[Union[int, Set[int]]] = None,
        eog_threshold: Optional[Union[float, str]] = _EOG_BLINK_THRESHOLD,
        use_eog: bool = True,
) -> mne.Epochs:
    """
    Detect blink onsets from eye-tracking and EOG channels and create epochs around each blink onset.
    By default, epochs are based on EOG detected blinks, using a threshold of 400μV. To prevent EOG blink detection, set
    the `eog_blink_threshold` parameter to None.
    To detect blinks from eye-tracking, provide the `et_channel` and `et_blink_codes` parameters. If both arguments are
    None, no eye-tracking blinks will be detected. If one is None and the other is not, a ValueError will be raised.

    :param raw: MNE Raw object containing the EEG data.
    :param sec_before: Number of seconds before each blink onset to include in the epoch.
    :param sec_after: Number of seconds after each blink onset to include in the epoch.
    :param baseline_sec: Tuple of (start, end) times for baseline correction relative to each blink onset.
    :param et_channel: Name of the eye-tracking channel in the raw data, or None to skip eye-tracking blink detection.
    :param et_blink_codes: Single or set of event codes that indicate a blink in the eye-tracking channel, or None to skip.
    :param eog_threshold: Threshold for detecting EOG blinks, 'auto' to use MNE's default threshold, or None to skip.
    :param use_eog: Whether to include EOG channels in the epoch data.

    :returns: MNE Epochs object containing the blink epochs.
    """
    assert sec_before >= 0 and sec_after >= 0, "Blink epoch times must be non-negative"
    new_raw = raw.copy()  # copy to avoid modifying the original raw data
    new_raw.set_annotations(
        # clear existing annotations to prevent blink detection from ignoring subset of the data
        mne.Annotations([], [], [])
    )
    new_raw.set_annotations(ah.blink_annotations(
        # detect blinks from eyetracking and EOG
        new_raw, et_channel, et_blink_codes, eog_threshold,
        0, 1, 1  # we only need blink onsets to create epochs
    ))
    blink_epochs = mne.Epochs(
        new_raw,
        tmin=-1 * sec_before, tmax=sec_after,
        baseline=baseline_sec or (-1 * sec_before, 0),
        picks=["eeg", "eog"] if use_eog else ["eeg"],
        events=None, preload=True, reject_by_annotation=False, reject=None,  # use annotations for epoching
        verbose=False
    )
    return blink_epochs
