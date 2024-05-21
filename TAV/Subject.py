import os
import warnings
from typing import Dict

import numpy as np
import pandas as pd
import pickle as pkl
import scipy as sp
import mne
from pymatreader import read_mat

import TAV.tav_helpers as tavh


class Subject:
    _BASE_PATH = r'C:\Users\nirjo\Desktop\TAV'
    # _BASE_PATH = r'E:\Tav'

    _REFERENCE_CHANNEL = "Pz"
    _PLOTTING_CHANNEL = "O2"

    _ET_EVENT_CODE = 1
    _FREE_VIEWING_START_CODE = 11
    _FREE_VIEWING_END_CODE = 12
    _SVP_START_CODE = 21
    _SVP_END_CODE = 22

    def __init__(self, idx: int):
        self.idx = idx
        # read data files
        self._eeg_no_eyemovements = self.__read_eeg_no_eyemovements(idx)  # all eye movements removed using ICA
        self._eeg_no_blinks = self.__read_eeg_no_blinks(idx)  # only blinks removed using ICA
        self._channels_map = self.__read_channels_map(idx)
        self._trial_data = self.__read_trial_data(idx)

        # extract useful information
        assert self._eeg_no_blinks.shape == self._eeg_no_eyemovements.shape
        self._num_channels, self._num_samples = self._eeg_no_eyemovements.shape
        trial_starts, trial_ends = self.__extract_trial_times(self._trial_data)  # trial start & end times
        ts = self.get_sample_indices()
        self._is_trial = ((ts >= trial_starts[:, None]) & (ts < trial_ends[:, None])).any(axis=0)
        # read events
        self._saccade_onset_idxs, self._saccade_offset_idxs = self.__extract_eye_tracker_events(self._trial_data)
        self._erp_idxs, self._frp_saccade_idxs, self._frp_fixation_idxs = self.__extract_event_related_potentials(self._trial_data)
        # calculate radial EOG
        self._reog_channel = self._calculate_radial_eog()

    @staticmethod
    def load_or_make(idx: int, dir_path: str = tavh.OUTPUT_DIR) -> 'Subject':
        try:
            with open(os.path.join(dir_path, f"Subject_{idx}.pkl"), 'rb') as f:
                loaded: Subject = pkl.load(f)
                return loaded
        except FileNotFoundError:
            with warnings.catch_warnings(action="ignore"):
                s = Subject(idx)
                s.to_pickle(dir_path)
                return s

    @property
    def num_samples(self) -> int:
        return self._num_samples

    @property
    def eeg_channels(self) -> Dict[str, int]:
        return {k: v for k, v in self._channels_map.items() if len(k) <= 3}

    @property
    def pickle_name(self) -> str:
        return f"Subject_{self.idx}.pkl"

    def to_pickle(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, self.pickle_name), 'wb') as f:
            pkl.dump(self, f)

    def get_sample_indices(self) -> np.ndarray:
        return np.arange(self.num_samples)

    def get_is_trial_channel(self) -> np.ndarray:
        return self._is_trial

    def get_channel(self, channel_name: str, full_ica: bool = False) -> np.ndarray:
        channel_name = channel_name.upper().strip()
        if channel_name == "REOG":
            return self._reog_channel
        if full_ica:
            return self._eeg_no_eyemovements[self._channels_map[channel_name]]
        return self._eeg_no_blinks[self._channels_map[channel_name]]

    def get_eye_tracking_event_indices(self, event: str, enforce_trials: bool = True) -> np.ndarray:
        event = event.upper().replace(" ", "_")
        if event in {"ERP"}:
            idxs = self._erp_idxs
            if enforce_trials:
                idxs = idxs[np.isin(idxs, np.where(self._is_trial)[0])]
            return idxs
        if event in {"FRP_SACCADE", "SACCADE_FRP"}:
            idxs = self._frp_saccade_idxs
            if enforce_trials:
                idxs = idxs[np.isin(idxs, np.where(self._is_trial)[0])]
            return idxs
        if event in {"FRP_FIXATION", "FIXATION_FRP"}:
            idxs = self._frp_fixation_idxs
            if enforce_trials:
                idxs = idxs[np.isin(idxs, np.where(self._is_trial)[0])]
            return idxs
        # make sure we return only the indices that are within trials, if enforce_trials is True
        if event in {"SACCADE_ONSET"} and not enforce_trials:
            return self._saccade_onset_idxs
        if event in {"SACCADE_OFFSET"} and not enforce_trials:
            return self._saccade_offset_idxs
        if event in {"SACCADE_ONSET", "SACCADE_OFFSET"} and enforce_trials:
            onset_idxs, offset_idxs = self._saccade_onset_idxs, self._saccade_offset_idxs
            matched_idxs = np.vstack([onset_idxs, offset_idxs]).T
            is_within_trial = np.isin(matched_idxs, np.where(self._is_trial)[0]).all(axis=1)
            col = 0 if event == "SACCADE_ONSET" else 1
            return matched_idxs[is_within_trial, col]
        raise ValueError(f"Event '{event}' not recognized")

    def calculate_reog_saccade_onset_indices(self,
                                             filter_name: str = 'srp',
                                             snr: float = 3.5,
                                             enforce_trials: bool = True) -> np.ndarray:
        """
        Detects saccade onsets in the radial EOG channel using the specified filter and signal-to-noise ratio.
        Returns an array of indices where saccade-onsets are detected from the radial EOG channel.
        """
        assert snr > 0, "Signal-to-noise ratio must be positive"
        filtered = tavh.apply_filter(self._reog_channel, filter_name)
        min_peak_height = filtered.mean() + snr * filtered.std()
        peak_idxs, _ = sp.signal.find_peaks(filtered, height=min_peak_height)
        if enforce_trials:
            peak_idxs = peak_idxs[np.isin(peak_idxs, np.where(self._is_trial)[0])]
        return peak_idxs

    def create_boolean_event_channel(self, event_idxs: np.ndarray, enforce_trials: bool = True) -> np.ndarray:
        """
        Creates a boolean array with length equal to the number of samples, where True values indicate the presence
        of an event at the corresponding index. If `enforce_trial` is True, only events that occur during a trial are
        marked as True.
        """
        is_event = tavh.create_boolean_array(self.num_samples, event_idxs)
        if enforce_trials:
            is_event &= self._is_trial
        return is_event

    def get_saccade_feature(
            self, feature: str, enforce_trials: bool = True
    ):
        feature = feature.lower().strip()
        if feature == "azimuth":
            col_name = "sac_angle"
        elif feature == "amplitude":
            col_name = "sac_amplitude"
        elif feature in {"vmax", "peak_velocity"}:
            col_name = "sac_vmax"
        else:
            raise ValueError(f"Feature '{feature}' not recognized")
        saccade_onset_idxs = self.get_eye_tracking_event_indices("saccade onset", enforce_trials)
        saccade_rows = self._trial_data[np.isin(self._trial_data["SacOnset"], saccade_onset_idxs)]
        return saccade_rows[col_name].values

    def plot_eyetracker_saccade_detection(self):
        # extract channels
        # TODO: add reog with butter/wavelet filters as well
        reog = self._reog_channel
        reog_srp_filtered = tavh.apply_filter(reog, "srp")
        is_et_saccade_channel = self.create_boolean_event_channel(self._saccade_onset_idxs, enforce_trials=False)

        # create mne object
        raw_object_data = np.vstack([self._reog_channel[self._is_trial],
                                     reog_srp_filtered[self._is_trial],
                                     is_et_saccade_channel[self._is_trial]])
        raw_object_info = mne.create_info(ch_names=['REOG', 'REOG_srp_filtered', 'ET_SACC'],
                                          ch_types=['eeg'] * 2 + ['stim'],
                                          sfreq=tavh.SAMPLING_FREQUENCY)
        raw_object = mne.io.RawArray(data=raw_object_data, info=raw_object_info)
        events = mne.find_events(raw_object, stim_channel='ET_SACC')
        scalings = dict(eeg=5e2, stim=1e10)
        fig = raw_object.plot(n_channels=2, events=events, scalings=scalings, event_color={1: 'r'}, show=False)
        fig.suptitle(f"ET Saccade Detection", y=0.99)
        fig.show()

    def _calculate_radial_eog(self) -> np.ndarray:
        mean_eog = np.nanmean(np.vstack([self._eeg_no_blinks[self._channels_map['LHEOG']],
                                         self._eeg_no_blinks[self._channels_map['RHEOG']],
                                         self._eeg_no_blinks[self._channels_map['RVEOGS']],
                                         self._eeg_no_blinks[self._channels_map['RVEOGI']]]),
                              axis=0)
        ref_channel = self._eeg_no_blinks[self._channels_map[Subject._REFERENCE_CHANNEL.upper()]]
        return mean_eog - ref_channel

    @staticmethod
    def __read_eeg_no_eyemovements(idx: int) -> np.ndarray:
        fname = os.path.join(Subject._BASE_PATH, "data", f"S{idx}_data_interp.mat")
        eeg_no_eyemovements = read_mat(fname)['data']
        eeg_no_eyemovements = np.swapaxes(eeg_no_eyemovements, 0, 1)
        return eeg_no_eyemovements

    @staticmethod
    def __read_eeg_no_blinks(idx: int) -> np.ndarray:
        fname = os.path.join(Subject._BASE_PATH, "data", f"S{idx}_data_ica_onlyBlinks.mat")
        eeg_no_blinks = read_mat(fname)['dat']
        eeg_no_blinks = np.swapaxes(eeg_no_blinks, 0, 1)
        return eeg_no_blinks

    @staticmethod
    def __read_channels_map(idx: int) -> Dict[str, int]:
        fname = os.path.join(Subject._BASE_PATH, "data", f"{idx}_channels.csv")
        df = pd.read_csv(fname, header=0, index_col=0)
        channels_series = df[df.columns[-1]]
        channels_dict = channels_series.to_dict()
        return {v.upper(): k for k, v in channels_dict.items()}

    @staticmethod
    def __read_trial_data(idx: int) -> pd.DataFrame:
        fname = os.path.join(Subject._BASE_PATH, "data", f"{idx}_info.csv")
        df = pd.read_csv(fname, header=0)
        return df

    @staticmethod
    def __extract_trial_times(trial_data: pd.DataFrame) -> (np.ndarray, np.ndarray):
        trial_start_times = trial_data[trial_data["Codes"] == Subject._FREE_VIEWING_START_CODE]['latency'].to_numpy()
        trial_end_times = trial_data[trial_data["Codes"] == Subject._FREE_VIEWING_END_CODE]['latency'].to_numpy()
        return trial_start_times, trial_end_times

    @staticmethod
    def __extract_eye_tracker_events(trial_data: pd.DataFrame) -> (np.ndarray, np.ndarray):
        is_et = trial_data["Codes"] == Subject._ET_EVENT_CODE
        saccade_onset_times = trial_data[is_et]['SacOnset'].to_numpy().astype('int64')
        saccade_durations = trial_data[is_et]['SacDur'].to_numpy().astype('int64')
        saccade_offset_times = saccade_onset_times + saccade_durations
        # TODO: assume inter-saccade intervals are fixations (are they?) and extract fixation onsets & offsets
        return saccade_onset_times, saccade_offset_times

    @staticmethod
    def __extract_event_related_potentials(trial_data: pd.DataFrame) -> (np.ndarray, np.ndarray, np.ndarray):
        is_erp = (trial_data["NewCodes"] // 10000 % 10 == 2) & (trial_data["NewCodes"] // 1000 % 10 == 0)
        erp_times = trial_data[is_erp]['latency'].to_numpy().astype('int64')
        is_frp = (
                (trial_data["NewCodes"] // 10000 % 10 == 1) &
                (trial_data["NewCodes"] // 1000 % 10 == 0) &
                (trial_data["NewCodes"] // 10 % 10 == 1)
        )
        frp_saccade_times = trial_data[is_frp]['SacOnset'].to_numpy().astype('int64')
        frp_fixation_times = trial_data[is_frp]['latency'].to_numpy().astype('int64')
        return erp_times, frp_saccade_times, frp_fixation_times
