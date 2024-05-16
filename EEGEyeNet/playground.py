import os

import numpy as np
import pandas as pd
import pickle as pkl

from pymatreader import read_mat


PATH = r'C:\Users\nirjo\Desktop\EEGEYENET\dots_data\synchronised_min\EP10\EP10_DOTS1_EEG.mat'

mat = read_mat(PATH)['sEEG']

num_channels = mat['nbchan']
num_samples = mat['pnts']
sampling_rate = mat['srate']
xmin, xmax = mat['xmin'], mat['xmax']

timestamps = mat['times']
data = mat['data']

chanlocs = mat['chanlocs']  # channel locations & labels
channel_labels = chanlocs['labels']

urchanlocs = mat['urchanlocs']
chaninfo = mat['chaninfo']
ref = mat['ref']

event = mat['event']  # event (triggers & ET events) information
urevent = mat['urevent']
event_df = pd.DataFrame(event)

etc = mat['etc']
