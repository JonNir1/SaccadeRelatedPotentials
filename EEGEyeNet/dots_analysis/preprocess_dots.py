import os
import pickle
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import mne
import matplotlib
import matplotlib.pyplot as plt
import easygui_qt.easygui_qt as gui

from EEGEyeNet.DataModels.Dots import DotsBlock

from mne_scripts.preprocess import preprocess_raw_fif
from mne_scripts.ica import run_ica

matplotlib.use('Qt5Agg')
mne.viz.set_browser_backend('qt')

# _BASE_PATH = r'C:\Users\jonathanni\Desktop\EEGEyeNet\dots_data\synchronised_min'  # lab
_BASE_PATH = r'C:\Users\nirjo\Desktop\SRP\data\EEGEyeNet\dots_data\synchronised_min'  # home

_SUBJ_ID = "EP12"
VISUALIZATION_SCALING = dict(eeg=1e-4, eog=1e-4, eyegaze=5e2, pupil=5e2)

# %%

# concat = mne.io.read_raw_fif(os.path.join(_BASE_PATH, _SUBJ_ID, "concatenated_raw.fif"), verbose=False)
# mne_events = mne.find_events(
#     concat,
#     stim_channel=pd.Series(concat.info['ch_names']).loc[pd.Series(concat.get_channel_types()) == 'stim'].tolist(),
#     output='onset',
#     shortest_event=1,
#     consecutive=True,
#     verbose=False
# )

preprocessed_raw = preprocess_raw_fif(
    os.path.join(_BASE_PATH, _SUBJ_ID, "concatenated_raw.fif"),
    eog_channels=DotsBlock.para_ocular_electrodes(),
    min_freq=0.1, max_freq=100, notch_freq=50,
    ref_channel="average", re_reference_eog=True,
    merge_within_ms=20,
    inspect_psd=True, block=True, interpolate_bads=False,
    verbose=True,
)

events_dict = pickle.load(open(os.path.join(_BASE_PATH, _SUBJ_ID, "events_dict.pkl"), 'rb'))
trial_onset_codes = {k: v for k, v in events_dict.items() if k.startswith('stim') and not k.endswith('off')}

cleaned_raw, ica = run_ica(
    preprocessed_raw,
    trial_events=trial_onset_codes,
    min_freq=2, epoch_with_eog=True, trial_reject_criteria=dict(eeg=100e-6, eog=250e-6),
    et_channel="STI_ET", et_blink_codes={215, 216}, eog_blink_threshold='auto', blink_epoch_repeats=1,
    num_components=25, random_state=42, max_iter=800, method='infomax', fit_params=dict(extended=True),
    ica_reject_criteria=dict(eeg=400e-6),
    plot_single_components=False, plot_cleaned_ica_psd=False, plot_cleaned_data=False,
    interpolate_bads=True,
    verbose=True,
)

num_components = ica.n_components
ica.plot_components(picks=range(num_components), title="IC Topo Maps", show=True)
ica.plot_sources(preprocessed_raw, title="IC Time Series", show=True)
# z = gui.get_list_of_choices(
#     title="Select components to exclude", choices=[f"Component {i:03d}" for i in range(20)]
# ) or []

for i in range(num_components):
    fig = ica.plot_properties(
        preprocessed_raw, picks=i, psd_args=dict(fmax=75), verbose=False, show=False
    )[0]
    fig.suptitle(f"Component {i + 1}/{num_components}")
    plt.show(block=False)
    # to_exclude = gui.get_continue_or_cancel(
    #     title=f"Exclude component {i}?",
    #     message="",
    #     continue_button_text="Exclude",
    #     cancel_button_text="Keep",
    # )
    plt.waitforbuttonpress()
    to_exclude = input(f"Exclude component {i}? (y/n): ").strip().lower() == 'y'
    # plt.close('all')
    if i == 1:
        break


# TODO: use the GUI to select components to exclude
# TODO: save ICA object, save pre-cleaning and post-cleaning raw, save events_dict
