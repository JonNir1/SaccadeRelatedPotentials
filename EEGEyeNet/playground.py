import os
from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd
import pickle as pkl
import mne

from EEGEyeNet.DataModels.DotsSession import DotsSession


PATH = r'C:\Users\jonathanni\Desktop\EEGEyeNet\dots_data\synchronised_min\EP12\EP12_DOTS3_EEG.mat'    #lab
# PATH = r'C:\Users\nirjo\Desktop\SRP\data\EEGEyeNet\dots_data\sunchronised_min\EP12\EP12_DOTS1_EEG.mat'  #home

ses = DotsSession.from_mat_file(PATH)
ts = ses.get_timestamps()
data = ses.get_data()
gaze_data = ses.get_gaze_data()
locs = ses.get_channel_locations()
labels = ses.get_channel_labels()
events = ses.get_events()
reog = ses.calculate_radial_eog('Pz')


raw, event_dict = ses.to_mne('Pz')

mne_events_et = mne.find_events(raw, stim_channel="STI_ET", output='onset', shortest_event=1, consecutive=True)
mne_events_ses = mne.find_events(raw, stim_channel="STI_SES", output='onset', shortest_event=1, consecutive=True)
mne_events_dot = mne.find_events(raw, stim_channel="STI_DOT", output='onset', shortest_event=1, consecutive=True)

# fig = mne.viz.plot_events(
#     mne_events_dot, sfreq=raw.info['sfreq'], event_id=event_dict, first_samp=raw.first_samp, on_missing='ignore'
# )

##############################################
# Epoching the data around stim onsets

import matplotlib
matplotlib.use('TkAgg')

dot_epochs = mne.Epochs(
    raw, mne_events_dot, event_id=event_dict, tmin=-0.4, tmax=1.0, preload=True, on_missing='ignore'
)
off_epochs = dot_epochs['stim/off']
on_epochs = dot_epochs[
    [key for key in dot_epochs.event_id.keys() if key.startswith('stim') and not key.endswith('off')]
]

on_epochs.plot(n_epochs=10, n_channels=5, events=False, block=True)

# TODO: check if there's scaling when generating the `raw` file and if so how to disable it
raw.plot(events=mne_events_dot, event_id=event_dict, n_channels=5, scalings=dict(eeg=1e1), block=True)


#########################################
# Contra-lateral Difference, following the analysis performed by Talcott et al., 2023 (https://doi.org/10.3758/s13414-023-02775-5)
# TODO: artifact/blink rejection or ICA - use synch_min or synch_max?
