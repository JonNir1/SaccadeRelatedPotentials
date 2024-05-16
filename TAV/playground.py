import os
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

import utils.visualization as vis
import TAV.tav_helpers as tavh
from TAV.Subject import Subject
from TAV.TavParticipant import TavParticipant
import TAV.window_sizes as window_sizes
import TAV.peri_saccade as peri_saccade
import TAV.signal_detection as signal_detection

pio.renderers.default = "browser"

# %%
#################################
# Window Sizes
subject_statistics, _, _ = window_sizes.load_or_calc()

# %%
#################################
# Peri-Saccade Reog Activity
subject_activities, _, _, _ = peri_saccade.load_or_calc_reog()

# %%
# SDT for Saccade Onset
subject_measures, subject_figures, mean_fig = signal_detection.load_or_calc_saccade_onset()


# %%
#################################
# Saccade Onset Differences

MAX_DIFF = 20

for idx in range(101, 111):
    s = Subject.load_or_make(idx, tavh.OUTPUT_DIR)
    et_onset_idxs = np.where(s.create_boolean_event_channel(s._saccade_onset_idxs, enforce_trials=True))[0]
    reog_onset_idxs = s.calculate_reog_saccade_onset_indices(filter_name='srp', snr=3.5, enforce_trials=True)

    onset_diffs = np.abs(et_onset_idxs - reog_onset_idxs.reshape((-1, 1)))
    onset_diffs_argmin = onset_diffs.argmin(axis=0).astype(float)   # index of best match
    onset_diffs_min = onset_diffs.min(axis=0).astype(float)         # minimum difference between ET and REOG onsets
    onset_diffs_argmin[onset_diffs_min > MAX_DIFF] = np.nan         # set to nan if difference is too large
    c = Counter(onset_diffs_argmin)
    print(f"Subject {idx}:\t{str(c.most_common(5))}")

# %%

idx = 101
s = Subject.load_or_make(idx, tavh.OUTPUT_DIR)
trial_data = s._trial_data

et_onset_idxs = np.where(s.create_boolean_event_channel(s._saccade_onset_idxs, enforce_trials=True))[0]
reog_onset_idxs = s.calculate_reog_saccade_onset_indices(filter_name='srp', snr=3.5, enforce_trials=True)
onset_diffs = np.abs(et_onset_idxs - reog_onset_idxs.reshape((-1, 1)))
onset_diffs_argmin = onset_diffs.argmin(axis=0).astype(float)  # index of best match
onset_diffs_min = onset_diffs.min(axis=0).astype(float)  # minimum difference between ET and REOG onsets
onset_diffs_argmin[onset_diffs_min > MAX_DIFF] = np.nan  # set to nan if difference is too large
match_idxs = np.stack([et_onset_idxs[~np.isnan(onset_diffs_argmin)],
                       reog_onset_idxs[onset_diffs_argmin[~np.isnan(onset_diffs_argmin)].astype(int)]]).T


table = abs(et_onset_idxs[:, None] - reog_onset_idxs[None, :])
rowwise = np.stack([table.argmin(0), np.arange(table.shape[1])]).T
colwise = np.stack([np.arange(table.shape[0]), table.argmin(1)]).T
rc = (rowwise[:, None] == colwise).all(-1).any(1)
idxs = rowwise[rc]
out = np.stack([et_onset_idxs[idxs[:, 0]], reog_onset_idxs[idxs[:, 1]]]).T

# TODO: apply this to `window_sizes.py`:
out = out[abs(np.diff(out, axis=1).flatten()) <= MAX_DIFF]


# %%
#################################
# Check how many ERPs exist and how many are during trials

idx = 101
s = Subject.load_or_make(idx, tavh.OUTPUT_DIR)

for idx in range(101, 111):
    s = Subject.load_or_make(idx, tavh.OUTPUT_DIR)
    num_erps = len(s._erp_idxs)
    num_erps_in_trials = sum(s.create_boolean_event_channel(s._erp_idxs, enforce_trials=True))
    print(f"Subject {idx}:\tERP onsets:\t{num_erps}\tIn Trials:\t{num_erps_in_trials}")

