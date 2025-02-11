import os

import numpy as np
import pandas as pd
import pickle as pkl
import mne

from pymatreader import read_mat

from EEGEyeNet.DataModels.DotsSession import DotsSession


PATH = r'C:\Users\jonathanni\Desktop\EEGEyeNet\dots_data\synchronised_min\EP12\EP12_DOTS5_EEG.mat'    #lab
# PATH = r'C:\Users\nirjo\Desktop\SRP\data\EEGEyeNet\dots_data\sunchronised_min\EP12\EP12_DOTS1_EEG.mat'  #home

mat = read_mat(PATH)['sEEG']

ses = DotsSession.from_mat_file(PATH)
ts = ses.get_timestamps()
data = ses.get_data()
gaze_data = ses.get_gaze_data()
locs = ses.get_channel_locations()
labels = ses.get_channel_labels()
events = ses.get_events()
eog = ses.get_eog_data()
eog_reog = ses.get_eog_data(reog_ref='Pz')

raw, event_dict = ses.to_mne()

mne_events_et = mne.find_events(raw, stim_channel="STI_ET", output='onset', shortest_event=1, consecutive=True)
mne_events_ses = mne.find_events(raw, stim_channel="STI_SES", output='onset', shortest_event=1, consecutive=True)
mne_events_stim = mne.find_events(raw, stim_channel="STI_DOT", output='onset', shortest_event=1, consecutive=True)

fig = mne.viz.plot_events(
    mne_events_ses, sfreq=raw.info['sfreq'], event_id=event_dict, first_samp=raw.first_samp, on_missing='ignore',
)
