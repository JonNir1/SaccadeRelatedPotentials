import os
import pickle
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import mne

import matplotlib
matplotlib.use('Qt5Agg')

from EEGEyeNet.DataModels.DotsSession import DotsSession
from mne_scripts.preprocess import preprocess_raw_fif


# _BASE_PATH = r'C:\Users\jonathanni\Desktop\EEGEyeNet\dots_data\synchronised_min'  # lab
_BASE_PATH = r'C:\Users\nirjo\Desktop\SRP\data\EEGEyeNet\dots_data\sunchronised_min'  # home
_MAT_FILE_FORMAT = "%s_DOTS%d_EEG.mat"

_SUBJ_ID = "EP12"
EEG_REF = 'Cz'
VISUALIZATION_SCALING = dict(eeg=1e-4, eog=1e-4, eyegaze=5e2, pupil=5e2)


# %%

def concatenate_sessions(base_path: str, subject_id: str) -> Tuple[mne.io.Raw, Dict[str, int]]:
    """ Concatenate all sessions of a subject, saves the concatenated raw and event dictionary to disk, and returns them. """
    if not os.path.exists(base_path):
        raise NotADirectoryError(base_path, "Base directory does not exist")
    subj_dir = os.path.join(base_path, subject_id)
    if not os.path.exists(subj_dir):
        raise NotADirectoryError(subj_dir, "Subject directory does not exist")
    concatenated_path = os.path.join(subj_dir, "concatenated_raw.fif")
    events_dict_path = os.path.join(subj_dir, "events_dict.pkl")
    try:
        concatenated = mne.io.read_raw_fif(concatenated_path, verbose=False)
        with open(events_dict_path, 'rb') as evnts_file:
            event_dict = pickle.load(evnts_file)
        return concatenated, event_dict
    except FileNotFoundError:
        raws = []
        for i in range(1, 7):
            mat_file = _MAT_FILE_FORMAT % (subject_id, i)
            mat_path = os.path.join(subj_dir, mat_file)
            raw, event_dict = DotsSession.from_mat_file(mat_path).to_mne()
            raws.append(raw)
            del raw, mat_path, mat_file
        concatenated = mne.concatenate_raws(raws, verbose=False)
        del raws
        concatenated.save(concatenated_path, picks="all", verbose=False)
        with open(events_dict_path, 'wb') as evnts_file:
            pickle.dump(event_dict, evnts_file, protocol=pickle.HIGHEST_PROTOCOL)
        return concatenated, event_dict


# %%

concat, events_dict = concatenate_sessions(_BASE_PATH, _SUBJ_ID)
mne_events = mne.find_events(
    concat,
    stim_channel=pd.Series(concat.info['ch_names']).loc[pd.Series(concat.get_channel_types()) == 'stim'].tolist(),
    output='onset',
    shortest_event=1,
    consecutive=True,
    verbose=False
)

preprocessed_raw = preprocess_raw_fif(
    os.path.join(_BASE_PATH, _SUBJ_ID, "concatenated_raw.fif"),
    eog_channels=DotsSession.para_ocular_electrodes(),
    min_freq=0.1, max_freq=100, notch_freq=50,
    ref_channel="average", re_reference_eog=True,
    merge_within_ms=20,
    inspect_psd=True, block=True, interpolate_bads=False,
    verbose=True,
)
