"""
A minimally working example to load and plot EEG data from an EEGEyeNet .mat file.
"""

import os

import numpy as np
import mne
from pymatreader import read_mat

import matplotlib
matplotlib.use('Qt5Agg')     # or 'TkAgg'

import matplotlib.pyplot as plt

_PARA_OCULAR_ELECTRODES = ['E25', 'E127', 'E8', 'E126', 'E32', 'E1', 'E17', 'E125', 'E128']
_ET_CHANNEL_MAP = {
    'L-GAZE-X': ('eyegaze', 'px', 'left', 'x'), 'L-GAZE-Y': ('eyegaze', 'px', 'left', 'y'),
    'R-GAZE-X': ('eyegaze', 'px', 'right', 'x'), 'R-GAZE-Y': ('eyegaze', 'px', 'right', 'y'),
    'L-AREA': ('pupil', 'au', 'left'), 'R-AREA': ('pupil', 'au', 'right'),
}



def load_eeg_eye_net(file_path: str, set_eog: bool = False) -> mne.io.Raw:
    """
    Load EEGEyeNet data from .mat file and convert to MNE Raw object.
    If `set_eog` is True, set the channel types for para-ocular electrodes as EOG channels
    """
    mat = read_mat(file_path)['sEEG']
    return eeg_eye_net_to_mne(mat, set_eog)


def eeg_eye_net_to_mne(mat, set_eog: bool = False) -> mne.io.Raw:
    """
    Convert EEGEyeNet data to MNE Raw object.
    If `set_eog` is True, set the channel types for para-ocular electrodes as EOG channels.
    """
    sfreq = mat['srate']
    data = mat['data']
    labels = mat['chanlocs']['labels']
    types = list(map(lambda chan: _channel_type(chan), labels))
    data[np.array(types) == 'eeg'] *= 1e-6      # convert uV to V on EEG channels
    info = mne.create_info(ch_names=labels, ch_types=types, sfreq=sfreq, verbose=False)
    raw = mne.io.RawArray(data, info, verbose=False)
    raw.set_montage("GSN-HydroCel-128", on_missing='ignore', verbose=False)

    # set channel types for eyetracking data
    is_eyetracking_channel = np.isin(raw.get_channel_types(), ['eyegaze', 'pupil'])
    eyetracking_channel_names = np.array(raw.ch_names)[is_eyetracking_channel]
    eyetracking_channel_mapping = {k: v for k, v in _ET_CHANNEL_MAP.items() if k in eyetracking_channel_names}
    raw = mne.preprocessing.eyetracking.set_channel_types_eyetrack(raw, mapping=eyetracking_channel_mapping)

    # set EOG channels
    if set_eog:
        raw.set_channel_types({ch: 'eog' for ch in _PARA_OCULAR_ELECTRODES})
    return raw


def _channel_type(chan: str):
    if chan.startswith('E') and chan[1:].isdigit():
        return 'eeg'
    if chan == 'Cz':
        return 'eeg'
    if 'GAZE' in chan:
        return 'eyegaze'
    if 'AREA' in chan:
        return 'pupil'
    return 'misc'

# %%

_SUBJECT_ID = "EP14"
_BASE_PATH = r'C:\Users\nirjo\Desktop\SRP\data\EEGEyeNet\dots_data\sunchronised_min'  # home
_FILE_PATH = f'{_SUBJECT_ID}\\{_SUBJECT_ID}_DOTS3_EEG.mat'
FULL_PATH = os.path.join(_BASE_PATH, _FILE_PATH)

raw = load_eeg_eye_net(FULL_PATH)
raw.set_eeg_reference("average", verbose=False)
spectrum = raw.copy().compute_psd(picks='eeg', n_fft=512, verbose=False)
# print((spectrum.data.min(), spectrum.data.max()))

fig = spectrum.plot(average=False, picks=['eeg'])
plt.show(block=False)      # need this to prevent figure from closing immediately
