import os
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import pickle as pkl
import mne

from pymatreader import read_mat


# PATH = r'C:\Users\jonathanni\Desktop\EEGEyeNet\dots_data\synchronised_min\EP12\EP12_DOTS5_EEG.mat'
PATH = r'C:\Users\nirjo\Desktop\SRP\data\EEGEyeNet\dots_data\sunchronised_min\EP12\EP12_DOTS1_EEG.mat'

mat = read_mat(PATH)['sEEG']

num_channels = mat['nbchan']
num_samples = mat['pnts']
sampling_rate = mat['srate']
xmin, xmax = mat['xmin'], mat['xmax']
ref = mat['ref']

timestamps = mat['times']
data = mat['data']      # channel data: num_channels x num_samples
gaze_data = data[129:]  # gaze data: 4 x num_samples (t, x, y, pupil) -> ignore t from this channel, use timestamps
is_missing_gaze_data = np.all(data[130:] == 0, axis=0)
data[130:, is_missing_gaze_data] = np.nan

chanlocs = mat['chanlocs']  # channel locations & labels
urchanlocs = mat['urchanlocs']
channel_labels = chanlocs['labels']
channel_types = chanlocs['type']

chaninfo = mat['chaninfo']  # not relevant


event = mat['event']  # event (triggers & ET events) information
event_df = pd.DataFrame(event)
event_df['type'] = event_df['type'].map(lambda val: val.strip()).map(lambda val: int(val) if val.isnumeric() else val)
event_df['block'] = 0

block_on = np.where(event_df['type'] == 55)[0]
block_off = np.where(event_df['type'] == 56)[0]
for i, (on, off) in enumerate(zip(block_on, block_off)):
    event_df.loc[on:off, 'block'] = i+1

urevent = mat['urevent']
urevent_df = pd.DataFrame(urevent).sort_values(by='latency')

reject = mat['reject']
stats = mat['stats']
etc = mat['etc']




counts = event_df['type'].map(lambda val: val.strip()).value_counts()
counts.index = counts.index.map(lambda val: int(val) if val.isnumeric() else val)
zcounts = counts[~np.isin(counts.index.to_list(), ['L_saccade', 'L_fixation', 'L_blink', 55, 56, 41])].sort_index()
print(zcounts.sum())    # should be 1+27*5 = 136 => block number (12/13/14/15/16) + dot number (1,2,...,27) * 5 blocks
