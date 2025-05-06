import os
import pickle
from numbers import Number

import numpy as np
import pandas as pd
import mne
import matplotlib
# import matplotlib.pyplot as plt

matplotlib.use('Qt5Agg')
# mne.viz.set_browser_backend('qt')

from EEGEyeNet.DataModels.Dots import DotsSession
import mne_scripts.helpers.raw_helpers as rh

# _BASE_PATH = r'C:\Users\jonathanni\Desktop\EEGEyeNet\dots_data\synchronised_min'  # lab
_BASE_PATH = r'C:\Users\nirjo\Desktop\SRP\data\EEGEyeNet\dots_data\synchronised_min'  # home

# EEG_REF = 'Cz'
VISUALIZATION_SCALING = dict(eeg=1e-4, eog=1e-4, eyegaze=5e2, pupil=5e2)


# %%
##############################################
# Load single DotsSession object and convert to MNE format

_SUBJ_ID = "EP10"

ses = DotsSession.from_mat(os.path.join(_BASE_PATH, _SUBJ_ID))
raw, events_dict = ses.to_mne()

raw = rh.remap_channels(raw, eog_channels=ses.get_block(1).para_ocular_electrodes())    # set EOG channels
channel_names = pd.Series(raw.info['ch_names'])
channel_types = pd.Series(raw.get_channel_types())

del _SUBJ_ID



b1 = ses.get_block(1)
evnts = b1.get_events()
onset_events = evnts.loc[evnts["type"].map(lambda t: isinstance(t, Number))]
onset_events = onset_events.drop(columns=[
    "sac_amplitude", "sac_endpos_x", "sac_endpos_y",
    "sac_startpos_x", "sac_startpos_y", "sac_vmax",
    "fix_avgpos_x", "fix_avgpos_y", "fix_avgpupilsize"
])
bins = np.linspace(-180, 180, num=3, endpoint=True)
bins = pd.cut(
    onset_events["prev_angle_deg"], bins=bins, labels=['r', 'l'],
).dropna().rename("prev_angle_interval")



# %%
##############################################
# Plot the concatenated data

stim_channel_names = channel_names.loc[pd.Series(raw.get_channel_types()) == 'stim'].tolist()
mne_events = mne.find_events(
    raw,
    stim_channel=stim_channel_names,
    output='onset',
    shortest_event=1,
    consecutive=True,
    verbose=False
)

# raw.plot(
#     block=True,
#     scalings=VISUALIZATION_SCALING, n_channels=15,
#     # events=mne_events
# )

del mne_events, stim_channel_names

# %%
##############################################
# Epoch trials based on stimulus-onset events (stim/1, stim/2, stim/3, ..., but not stim/41 which is the offset)

DOT_STIM_CHANNEL = "STI_DOT"
EPOCH_BEFORE_STIM, EPOCH_AFTER_STIM = -0.3, 1.0     # seconds
BASELINE_START, BASELINE_END = -0.25, -0.05         # seconds

stim_events = mne.find_events(
    raw,
    stim_channel=DOT_STIM_CHANNEL,
    output='onset',
    shortest_event=1,
    consecutive=True,
    verbose=False
)
stim_epochs = mne.Epochs(
    raw, stim_events, event_id=events_dict,
    tmin=EPOCH_BEFORE_STIM, tmax=EPOCH_AFTER_STIM,
    baseline=(BASELINE_START, BASELINE_END),
    reject_by_annotation=True, preload=True, on_missing='ignore', verbose=False,
)

stim_onset_epochs = stim_epochs[[
    key for key in stim_epochs.event_id.keys() if key.startswith('stim') and not key.endswith('off')
]]

stim_onset_averaged = stim_onset_epochs.average(picks='eeg')
stim_onset_averaged.plot(picks=["E65", "E90"], spatial_colors=True, time_unit='s', show=True)

del DOT_STIM_CHANNEL, EPOCH_BEFORE_STIM, EPOCH_AFTER_STIM, BASELINE_START, BASELINE_END, stim_events, stim_epochs

# %%
##############################################
# TODO: visualize activity in PO7 (E65) and PO8 (E90) for each epoch
# EGI to 10-20 mapping: https://www.egi.com/images/HydroCelGSN_10-10.pdf

eog_channel_names = channel_names.loc[channel_types == 'eog'].tolist()
gaze_channel_names = channel_names.loc[channel_types == 'eyegaze'].tolist()
stim_onset_epochs_fig = stim_onset_epochs.plot(
    n_epochs=5, n_channels=20, events=True, scalings=VISUALIZATION_SCALING,
    picks=['E65', 'E90'] + eog_channel_names + gaze_channel_names,
    block=True,
)

# TODO: calculate difference of PO7-PO8 activity for each epoch
#  check if N2pc exists and if it precedes the saccade
#  regress N2pc amplitude/latency with saccade amplitude and latency

#########################################
# Contra-lateral Difference, following the analysis performed by Talcott et al., 2023 (https://doi.org/10.3758/s13414-023-02775-5)
# TODO: artifact/blink rejection or ICA - use synch_min or synch_max?
