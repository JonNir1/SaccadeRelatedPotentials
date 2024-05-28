import os
import time
import warnings
from typing import List
from collections import Counter

import numpy as np
import pandas as pd
import pickle as pkl
from pymatreader import read_mat
import scipy.signal as signal
import pywt
from tqdm import tqdm
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

import TAV.tav_helpers as tavh
from TAV.Subject import Subject

pio.renderers.default = "browser"

# %%
#################################
# Window Sizes
import TAV.window_sizes as window_sizes
subject_statistics, _, _ = window_sizes.load_or_calc_saccade_onset()

# %%
#################################
# SDT for Saccade Onset
import TAV.signal_detection as signal_detection
subject_measures, _, _ = signal_detection.load_or_calc_saccade_onset()

# %%
#################################
# playground
import TAV.timing_differences as time_diffs
diffs, _, _ = time_diffs.load_or_calc_saccade_timing_differences()

# %%
#################################
# Peri-Saccades
import TAV.peri_saccade as peri_saccade
subject_epochs, _, _ = peri_saccade.load_or_calc()

# %%
#################################
# saccade epochs by direction

start = time.time()

import TAV.peri_saccade as peri_saccade

output_dir = os.path.join(r"C:\Users\nirjo\Desktop\SRP", "peri_saccade_data")
os.makedirs(output_dir)

for idx in tqdm(range(101, 111)):
    s = Subject.load_or_make(idx, tavh.RESULTS_DIR)
    azimuth = s.get_saccade_feature("azimuth", enforce_trials=True)
    is_rightwards = (-90 <= azimuth) & (azimuth < 90)
    epochs = peri_saccade.load_or_calc_epochs(s)
    right_epochs = epochs.map(lambda cell: cell[is_rightwards] if cell is pd.DataFrame else cell)
    left_epochs = epochs.map(lambda cell: cell[~is_rightwards] if cell is pd.DataFrame else cell)
    left = {
        ("left", event, channel): channel_values
        for event, event_values in left_epochs.to_dict().items()
        for channel, channel_values in event_values.items()
    }
    right = {
        ("right", event, channel): channel_values
        for event, event_values in right_epochs.to_dict().items()
        for channel, channel_values in event_values.items()
    }
    both = {**left, **right}
    path = os.path.join(output_dir, f"s{idx}")
    os.makedirs(path, exist_ok=True)
    for key, df in tqdm(both.items()):
        if df is not None and not np.isnan(df):
            fname = "_".join(key) + ".csv"
            df.to_csv(os.path.join(path, fname))
    # todo: save as matlab struct

del azimuth, is_rightwards, epochs, left_epochs, right_epochs, left, right, both, path, fname

elapsed = time.time() - start

# %%
#################################
# Check how many ERPs exist and how many are during trials

idx = 101
s = Subject.load_or_make(idx, tavh.RESULTS_DIR)

for idx in range(101, 111):
    s = Subject.load_or_make(idx, tavh.RESULTS_DIR)
    num_erps = len(s._erp_idxs)
    num_erps_in_trials = sum(s.create_boolean_event_channel(s._erp_idxs, enforce_trials=True))
    print(f"Subject {idx}:\tERP onsets:\t{num_erps}\tIn Trials:\t{num_erps_in_trials}")
