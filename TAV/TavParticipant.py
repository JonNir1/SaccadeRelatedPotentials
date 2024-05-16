import os
import sys
import mne
import numpy as np
from pymatreader import read_mat
from scipy.signal import find_peaks
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import savemat
from scipy import stats, signal

mne.set_log_level(verbose="CRITICAL")
mne.viz.set_3d_backend("notebook")

plots_dir = "plots"
channel_for_plots = "O2"
channels_for_epochs = ["O1", "O2", "P10"]

####################################################################

SRPfilt = [0, -0.0000, -0.0001, -0.0002, -0.0002, -0.0001, 0.0001, 0.0003, 0.0007, 0.0015, 0.0028,
           0.0050, 0.0080, 0.0114, 0.0151, 0.0188, 0.0217, 0.0241, 0.0267, 0.0272, 0.0271, 0.0287,
           0.0329, 0.0391, 0.0462, 0.0544, 0.0605, 0.0602, 0.0447, 0.0030, -0.0672, -0.1615, -0.2631,
           -0.3490, -0.3965, -0.3834, -0.3045, -0.1706, -0.0109, 0.1349, 0.2355, 0.2789, 0.2707, 0.2271,
           0.1683, 0.1100, 0.0631, 0.0319, 0.0174, 0.0142, 0.0193, 0.0274, 0.0312, 0.0303, 0.0257,
           0.0183, 0.0088, -0.0007, -0.0086, -0.0152, -0.0198, -0.0221, -0.0229, -0.0230, -0.0219, -0.0199,
           -0.0179, -0.0157, -0.0129, -0.0101, -0.0070, -0.0042, -0.0020, -0.0003, 0.0009, 0.0013, 0.0013,
           0.0011, 0.0008, 0.0005, 0.0002, 0.0001, 0.0000, 0]


def filterSRP(x, fs=1024):
    # resample to 1024 if necessary
    if fs != 1024:
        x = resmaple(x, fs, 1024)
    # Number of time-samples in the filter
    n = len(SRPfilt)
    SPonset = 28

    x = np.convolve(x, SRPfilt[::-1])
    x = x[n - SPonset: -SPonset + 2]

    # resample back if necessary
    if fs != 1024:
        x = resmaple(x, 1024, fs)

    return x


####################################################################

class TavParticipant:
    # _BASE_PATH = r'E:\Tav'
    _BASE_PATH = r'C:\Users\nirjo\Desktop\TAV'

    def __init__(self, participant):
        self.participant = participant

        self.only_blinks_data = \
            read_mat(os.path.join(TavParticipant._BASE_PATH, "data", f"S{self.participant}_data_ica_onlyBlinks.mat"))[
                'dat']
        self.r = np.arange(self.only_blinks_data.shape[0])

        self.only_blinks_data = np.swapaxes(self.only_blinks_data, 0, 1)

        self.interp_data = \
        read_mat(os.path.join(TavParticipant._BASE_PATH, "data", f"S{self.participant}_data_interp.mat"))['data']
        self.interp_data = np.swapaxes(self.interp_data, 0, 1)
        self.channels = pd.read_csv(os.path.join(TavParticipant._BASE_PATH, "data", f"{self.participant}_channels.csv"))

        self.info_data = pd.read_csv(os.path.join(TavParticipant._BASE_PATH, "data", f"{self.participant}_info.csv"))

        self.start_idx = self.info_data[self.info_data["Codes"] == 11]['latency'].to_numpy()
        self.end_idx = self.info_data[self.info_data["Codes"] == 12]['latency'].to_numpy()
        self.trials_length = np.cumsum(self.end_idx - self.start_idx)

        self.epochs_erp_sacc_fname = os.path.join(TavParticipant._BASE_PATH, 'epochs',
                                                  f'{self.participant}-erp-sacc-epo.fif')
        self.epochs_frp_et_sacc_fname = os.path.join(TavParticipant._BASE_PATH, 'epochs',
                                                     f'{self.participant}-frp-et-sacc-epo.fif')
        self.epochs_frp_et_fix_fname = os.path.join(TavParticipant._BASE_PATH, 'epochs',
                                                    f'{self.participant}-frp-et-fix-epo.fif')
        self.epochs_frp_eog_sacc_fname = os.path.join(TavParticipant._BASE_PATH, 'epochs',
                                                      f'{self.participant}-frp-eog-sacc-epo.fif')

        self.channel_for_plots = "O2"

        self.calc_eye_tracker_sacc()

        self.calc_reog_channel()

        self.calc_eog_sacc()

        self.calc_events()

        self.get_epochs()

    def plot_saccade_detection(self):
        mask = ((self.r >= self.start_idx[:, None]) & (self.r < self.end_idx[:, None])).any(0)
        raw_object_data = np.vstack([self.reog_channel[mask], self.SRPed_data[mask], self.eye_tracker_sacc_vec[mask]])
        raw_object_ch = ['REOG', 'REOG_filtered', 'ET_SACC']
        raw_object_ch_types = ['eeg'] * 2 + ['stim']
        raw_object_info = mne.create_info(raw_object_ch, sfreq=1024, ch_types=raw_object_ch_types)
        raw_object = mne.io.RawArray(data=raw_object_data,
                                     info=raw_object_info)
        events = mne.find_events(raw_object, stim_channel='ET_SACC')
        scalings = dict(eeg=5e2, stim=1e10)
        fig = raw_object.plot(n_channels=2, events=events, scalings=scalings, event_color={1: 'r'}, show=False)
        fig.suptitle(f"Figure 1", y=1.01)
        plt.show()

    def ch_idx(self, channel):
        return self.channels.index[self.channels['T'] == channel].item()

    def filter_trials(self, channel):
        return channel[((self.r >= self.start_idx[:, None]) & (self.r < self.end_idx[:, None])).any(0)]

    def calc_eye_tracker_sacc(self):
        self.eye_tracker_sacc_idx = self.info_data[self.info_data["Codes"] == 1]['SacOnset'].to_numpy().astype('int64')
        self.eye_tracker_sacc_vec = np.zeros_like(self.only_blinks_data[0, :])
        np.put(self.eye_tracker_sacc_vec, ind=self.eye_tracker_sacc_idx, v=1)

    def calc_reog_channel(self):

        self.reog_channel = (self.only_blinks_data[self.ch_idx('LHEOG'), :] + self.only_blinks_data[
                                                                              self.ch_idx('RHEOG'), :] +
                             self.only_blinks_data[self.ch_idx('RVEOGS'), :] + self.only_blinks_data[
                                                                               self.ch_idx('RVEOGI'), :]) \
                            / 4 - self.only_blinks_data[self.ch_idx('Pz'), :]

    def calc_eog_sacc(self, std_factor=3.5, plot=False):
        self.SRPed_data = filterSRP(self.reog_channel, 1024)[:-1]

        mean = self.SRPed_data.mean()
        std = self.SRPed_data.std()

        above_threshold_idx = find_peaks(self.SRPed_data, height=mean + std_factor * std)[0]
        above_threshold = np.zeros_like(self.SRPed_data)
        np.put(above_threshold, ind=above_threshold_idx, v=1)

        mask = ((self.r >= self.start_idx[:, None]) & (self.r < self.end_idx[:, None])).any(0)

        self.above_threshold = np.zeros_like(self.SRPed_data)
        self.above_threshold[mask] = above_threshold[mask]

        self.above_threshold_idx = np.squeeze(np.argwhere(self.above_threshold))

    def calc_events(self):
        self.eye_tracker_erp_idx = self.info_data[(self.info_data["NewCodes"] // 10000 % 10 == 2) &
                                                  (self.info_data["NewCodes"] // 1000 % 10 == 0)][
            'latency'].to_numpy().astype('int64')
        self.eye_tracker_erp_vec = np.zeros_like(self.only_blinks_data[0, :])
        np.put(self.eye_tracker_erp_vec, ind=self.eye_tracker_erp_idx, v=1)

        self.eye_tracker_frp_sacc_idx = self.info_data[(self.info_data["NewCodes"] // 10000 % 10 == 1) &
                                                       (self.info_data["NewCodes"] // 10 % 10 == 1) &
                                                       (self.info_data["NewCodes"] // 1000 % 10 == 0)][
            'SacOnset'].to_numpy().astype('int64')
        self.eye_tracker_frp_sacc_vec = np.zeros_like(self.only_blinks_data[0, :])
        np.put(self.eye_tracker_frp_sacc_vec, ind=self.eye_tracker_frp_sacc_idx, v=1)

        self.eye_tracker_frp_fix_idx = self.info_data[(self.info_data["NewCodes"] // 10000 % 10 == 1) &
                                                      (self.info_data["NewCodes"] // 10 % 10 == 1) &
                                                      (self.info_data["NewCodes"] // 1000 % 10 == 0)][
            'latency'].to_numpy().astype('int64')
        self.eye_tracker_frp_fix_vec = np.zeros_like(self.only_blinks_data[0, :])
        np.put(self.eye_tracker_frp_fix_vec, ind=self.eye_tracker_frp_fix_idx, v=1)

        self.eog_frp_idx = self.above_threshold_idx
        self.eog_frp_vec = np.zeros_like(self.only_blinks_data[0, :])
        np.put(self.eog_frp_vec, ind=self.eog_frp_idx, v=1)

    def calc_epochs(self):

        info = mne.create_info(ch_names=["O2", "O1", "P10", "ERP", "FRP_ET_SACC", "FRP_ET_FIX", "FRP_EOG"], sfreq=1024,
                               ch_types=['eeg', 'eeg', 'eeg', 'stim', 'stim', 'stim', 'stim'])
        data = self.interp_data[[self.ch_idx("O2"), self.ch_idx("O1"), self.ch_idx("P10")], :]

        data = np.vstack([data, self.eye_tracker_erp_vec, self.eye_tracker_frp_sacc_vec, self.eye_tracker_frp_fix_vec,
                          self.eog_frp_vec])

        raw = mne.io.RawArray(data=data, info=info)

        erp_events = mne.find_events(raw=raw, stim_channel="ERP")
        frp_et_sacc_events = mne.find_events(raw=raw, stim_channel="FRP_ET_SACC")
        frp_et_fix_events = mne.find_events(raw=raw, stim_channel="FRP_ET_FIX")
        frp_eog_events = mne.find_events(raw=raw, stim_channel="FRP_EOG")

        self.erp_epochs = mne.Epochs(raw=raw, events=erp_events, tmin=-0.2, tmax=0.6, preload=True)
        self.erp_epochs.save(fname=self.epochs_erp_sacc_fname, overwrite=True)

        self.frp_et_sacc_epochs = mne.Epochs(raw=raw, events=frp_et_sacc_events, tmin=-0.2, tmax=0.6, preload=True)
        self.frp_et_sacc_epochs.save(fname=self.epochs_frp_et_sacc_fname, overwrite=True)

        self.frp_et_fix_epochs = mne.Epochs(raw=raw, events=frp_et_fix_events, tmin=-0.2, tmax=0.6, preload=True)
        self.frp_et_fix_epochs.save(fname=self.epochs_frp_et_fix_fname, overwrite=True)

        self.frp_eog_epochs = mne.Epochs(raw=raw, events=frp_eog_events, tmin=-0.2, tmax=0.6, preload=True)
        self.frp_eog_epochs.save(fname=self.epochs_frp_eog_sacc_fname, overwrite=True)

        epochs = {"ERP": self.erp_epochs, "FRP_ET_SACC": self.frp_et_sacc_epochs, "FRP_ET_FIX": self.frp_et_fix_epochs,
                  "FRP_EOG": self.frp_eog_epochs}
        mat_data = {"O1": {}, "O2": {}, "P10": {}}
        for channel in channels_for_epochs:
            for kind in epochs.keys():
                mat_data[channel][kind] = np.squeeze(epochs[kind].get_data(picks=channel))
        mdict = {"data": mat_data}
        with open(f"S{self.participant}_epochs.mat", 'wb') as f:
            savemat(f, mdict)

    def get_epochs(self):
        if os.path.isfile(self.epochs_erp_sacc_fname) and os.path.isfile(self.epochs_frp_et_sacc_fname) \
                and os.path.isfile(self.epochs_frp_eog_sacc_fname) and os.path.isfile(self.epochs_frp_et_fix_fname):
            self.erp_epochs = mne.read_epochs(fname=self.epochs_erp_sacc_fname)
            self.frp_et_sacc_epochs = mne.read_epochs(fname=self.epochs_frp_et_sacc_fname)
            self.frp_et_fix_epochs = mne.read_epochs(fname=self.epochs_frp_et_fix_fname)
            self.frp_eog_epochs = mne.read_epochs(fname=self.epochs_frp_eog_sacc_fname)
        else:
            self.calc_epochs()

    def count_hits(self):
        as_strided = np.lib.stride_tricks.as_strided
        r = 9
        above_threshold_ext = np.concatenate((np.full(r, np.nan), self.above_threshold, np.full(r, np.nan)))
        windows = as_strided(above_threshold_ext,
                             (above_threshold_ext.shape[0], 2 * r + 1),
                             above_threshold_ext.strides * 2)
        windows = windows[self.eye_tracker_sacc_idx]
        detected = np.count_nonzero(np.sum(windows, axis=1))
        sacc_count = np.sum(self.eye_tracker_sacc_vec)  # len(s101._saccade_onset_idxs)
        ## p101.above_threshold is s101.is_reog_saccade_onset
        ## p101.eye_tracker_sacc_idx is s101._saccade_onset_idxs
        return detected / sacc_count

    def count_false_alarms(self):
        section_len = 19
        N = self.above_threshold.shape[0] // section_len * section_len

        sacc_windows = np.reshape(self.eye_tracker_sacc_vec[:N],
                                  (-1, section_len))  # eye_tracker_sacc_vec is for entire session
        above_threshold_windows = np.reshape(self.above_threshold[:N],
                                             (-1, section_len))  # above_threshold is during trials only!

        is_sacc = sacc_windows.sum(axis=1)
        count_no_sacc = np.count_nonzero(is_sacc == 0)

        is_sp = above_threshold_windows.sum(axis=1)
        is_sp = np.where(is_sacc == 0, is_sp, 0)
        count_sp_no_sacc = np.count_nonzero(is_sp)

        return count_sp_no_sacc / count_no_sacc
