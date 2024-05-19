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
#################################
# SDT for Saccade Onset
subject_measures, _, _ = signal_detection.load_or_calc_saccade_onset()

# %%
#################################
# playground
_SAMPLES_BEFORE, _SAMPLES_AFTER = 250, 250

idx = 101
s = Subject.load_or_make(idx, tavh.OUTPUT_DIR)
trial_data = s._trial_data
F4, F3 = s.get_eeg_channel("F4", False), s.get_eeg_channel("F3", False)
diff = F4 - F3
# diff[~s.get_is_trial_channel()] = np.nan
# TODO: change the sign of `diff` according to the direction of the saccade

# is_in_saccade = np.full_like(diff, False)
sac_onsets = s.get_eye_tracking_event_indices("saccade onset")
sac_offsets = s.get_eye_tracking_event_indices("saccade offset")
# for onset, offset in zip(sac_onsets, sac_offsets):
#     is_in_saccade[onset:offset] = True
# is_in_saccade[~s.get_is_trial_channel()] = False

saccade_direction = np.full_like(diff, None, dtype=str)
for onset, offset in zip(sac_onsets, sac_offsets):
    azimuth = trial_data[trial_data["SacOnset"] == onset]["sac_angle"].values[0]
    saccade_direction[onset:offset] = "R" if -90 <= azimuth <= 90 else "L"
# saccade_direction[~s.get_is_trial_channel()] = None

diff[saccade_direction == "L"] = -diff[saccade_direction == "L"]

###########
is_et_saccade_on_channel = s.create_boolean_event_channel(
    s.get_eye_tracking_event_indices("saccade onset"), enforce_trials=True
)
et_on_epochs = peri_saccade._peri_event_activity(
    is_event=is_et_saccade_on_channel, channel=diff, n_samples_before=_SAMPLES_BEFORE, n_samples_after=_SAMPLES_AFTER
)
et_on_fig = peri_saccade._peri_event_line_figure(
    pd.Series({"diff": et_on_epochs}, name="ET Onset"), f"Subject {idx}", show_error=False
)
et_on_fig.show()

###########
is_reog_saccade_on_channel = s.create_boolean_event_channel(
    s.calculate_reog_saccade_onset_indices(), enforce_trials=True
)
reog_on_epochs = peri_saccade._peri_event_activity(
    is_event=is_reog_saccade_on_channel, channel=diff, n_samples_before=_SAMPLES_BEFORE, n_samples_after=_SAMPLES_AFTER
)
reog_on_fig = peri_saccade._peri_event_line_figure(
    pd.Series({"diff": reog_on_epochs}, name="REOG Onset"), f"Subject {idx}", show_error=False
)
reog_on_fig.show()

###########
is_et_saccade_off_channel = s.create_boolean_event_channel(
    s.get_eye_tracking_event_indices("saccade offset"), enforce_trials=True
)
et_off_epochs = peri_saccade._peri_event_activity(
    is_event=is_et_saccade_off_channel, channel=diff, n_samples_before=_SAMPLES_BEFORE, n_samples_after=_SAMPLES_AFTER
)
et_off_fig = peri_saccade._peri_event_line_figure(
    pd.Series({"diff": et_off_epochs}, name="ET Offset"), f"Subject {idx}", show_error=False
)
et_off_fig.show()

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

