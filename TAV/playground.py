
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
# Saccade Epochs
import TAV.saccade_epochs as saccade_epochs

epochs, _ = saccade_epochs.load_or_calc()

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
