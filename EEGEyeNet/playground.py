import os

import numpy as np
import pandas as pd
import pickle as pkl
import mne

from pymatreader import read_mat


PATH = r'C:\Users\jonathanni\Desktop\EEGEyeNet\dots_data\synchronised_min\EP12\EP12_DOTS5_EEG.mat'

mat = read_mat(PATH)['sEEG']

num_channels = mat['nbchan']
num_samples = mat['pnts']
sampling_rate = mat['srate']
xmin, xmax = mat['xmin'], mat['xmax']

timestamps = mat['times']
data = mat['data']      # channel data: num_channels x num_samples

chanlocs = mat['chanlocs']  # channel locations & labels
channel_labels = chanlocs['labels']

urchanlocs = mat['urchanlocs']
chaninfo = mat['chaninfo']
ref = mat['ref']

event = mat['event']  # event (triggers & ET events) information
event_df = pd.DataFrame(event)

urevent = mat['urevent']
urevent_df = pd.DataFrame(urevent).sort_values(by='latency')


reject = mat['reject']
stats = mat['stats']
etc = mat['etc']




counts = event_df['type'].value_counts()
counts.index = counts.index.map(lambda val: val.strip())
zcounts = counts[~np.isin(counts.index, ['L_saccade', 'L_fixation', 'L_blink', 55, 56, 41])]
print(zcounts.sum())    # should be 1+27*5 = 136 => block number (12/13/14/15/16) + dot number (1,2,...,27) * 5 blocks
