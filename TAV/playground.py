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
subject_measures, subject_figures, mean_fig = signal_detection.load_or_calc_saccade_onset()

# %%
#################################
# playground

idx = 101
s = Subject.load_or_make(idx, tavh.OUTPUT_DIR)
trial_data = s._trial_data

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

