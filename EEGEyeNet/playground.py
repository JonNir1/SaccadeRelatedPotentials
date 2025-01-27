import os

import numpy as np
import pandas as pd
import pickle as pkl
import mne

from pymatreader import read_mat

from EEGEyeNet.DataModels.Session import DotsSession


PATH = r'C:\Users\jonathanni\Desktop\EEGEyeNet\dots_data\synchronised_min\EP12\EP12_DOTS5_EEG.mat'
# PATH = r'C:\Users\nirjo\Desktop\SRP\data\EEGEyeNet\dots_data\sunchronised_min\EP12\EP12_DOTS1_EEG.mat'

mat = read_mat(PATH)['sEEG']


ses = DotsSession.from_mat_file(PATH)
data = ses.get_data()
ts = ses.get_timestamps()
locs = ses.get_channel_locations()
events = ses.get_events()


###############################
# TODO: Create MNE Raw object #

# convert "events" DF to MNE trigger channel and add it to the Info+Raw objects

info = mne.create_info(ch_names=ses.get_channel_labels().tolist(), sfreq=ses.sampling_rate, ch_types=locs['type'])
raw = mne.io.RawArray(data, info)

#######################
# TODO: add methods to extract gaze data
