import os

import numpy as np
import pandas as pd
import pickle as pkl
import mne

from pymatreader import read_mat

from EEGEyeNet.DataModels.DotsSession import DotsSession


# PATH = r'C:\Users\jonathanni\Desktop\EEGEyeNet\dots_data\synchronised_min\EP12\EP12_DOTS5_EEG.mat'
PATH = r'C:\Users\nirjo\Desktop\SRP\data\EEGEyeNet\dots_data\sunchronised_min\EP12\EP12_DOTS1_EEG.mat'

mat = read_mat(PATH)['sEEG']

ses = DotsSession.from_mat_file(PATH)
ts = ses.get_timestamps()
data = ses.get_data()
gaze_data = ses.get_gaze_data()
locs = ses.get_channel_locations()
events = ses.get_events()
raw = ses.to_mne()
