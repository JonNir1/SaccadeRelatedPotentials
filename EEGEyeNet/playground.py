import os
from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd
import pickle as pkl
import mne
import matplotlib

from utils import mne_helpers as mnh
from EEGEyeNet.DataModels.DotsSession import DotsSession

# matplotlib.use('TkAgg')

PATH = r'C:\Users\jonathanni\Desktop\EEGEyeNet\dots_data\synchronised_min\EP12\EP12_DOTS3_EEG.mat'    #lab
# PATH = r'C:\Users\nirjo\Desktop\SRP\data\EEGEyeNet\dots_data\sunchronised_min\EP12\EP12_DOTS1_EEG.mat'  #home

VISUALIZATION_SCALING = dict(eeg=1e-4, eog=1e-4, eyegaze=5e2, pupil=5e2)

# %%
##############################################
# Loa the data and convert to MNE format

ses = DotsSession.from_mat_file(PATH)
# ts = ses.get_timestamps()
# data = ses.get_data()
# gaze_data = ses.get_gaze_data()
# locs = ses.get_channel_locations()
# labels = ses.get_channel_labels()
# events = ses.get_events()
# reog = ses.calculate_radial_eog(reog_ref='Pz')

raw, event_dict = ses.to_mne(reog_ref='Pz')

# raw.plot(block=True, scalings=VISUALIZATION_SCALING, n_channels=5)

# %%
##############################################
# Filter

NOTCH_FREQ, LOW_FREQ, HIGH_FREQ = 50, 0.1, 100

raw.notch_filter(freqs=NOTCH_FREQ, fir_design='firwin', picks=["eeg", "eog"])               # remove AC line noise
raw.filter(l_freq=LOW_FREQ, h_freq=HIGH_FREQ, fir_design='firwin', picks=["eeg", "eog"])    # band-pass filter


# %%
##############################################
# Detect and Annotate Blinks

BEFORE_BLINK, AFTER_BLINK = 25, 25  # annotate 25ms before & after each detected blink

blink_annots_et = mnh.eyetracking_blink_annotations(raw, 'STI_ET', {215, 216}, BEFORE_BLINK, AFTER_BLINK)
blink_annots_eog = mnh.eog_blink_annotations(raw, BEFORE_BLINK, AFTER_BLINK)
raw.set_annotations(blink_annots_et + blink_annots_eog)

# raw.plot(block=True, scalings=VISUALIZATION_SCALING, n_channels=5)

# %%
##############################################
# Epoch trials based on `stim/{%d}` events

BEFORE_EPOCH_SEC, AFTER_EPOCH_SEC = 0.5, 1.5

mne_dot_events = mne.find_events(raw, stim_channel="STI_DOT", output='onset', shortest_event=1, consecutive=True)
dot_epochs = mne.Epochs(
    raw, mne_dot_events, event_id=event_dict,
    tmin=-1 * BEFORE_EPOCH_SEC, tmax= AFTER_EPOCH_SEC,
    reject_by_annotation=True, preload=True, on_missing='ignore'
)
off_epochs = dot_epochs['stim/off']
on_epochs = dot_epochs[
    [key for key in dot_epochs.event_id.keys() if key.startswith('stim') and not key.endswith('off')]
]

# mne.viz.plot_events(
#     mne_dot_events, sfreq=raw.info['sfreq'], event_id=event_dict,
#     first_samp=raw.first_samp, on_missing='ignore',
# )

# on_epochs.plot(block=True, n_epochs=10, n_channels=5, events=False, scalings=VISUALIZATION_SCALING)

# %%
##############################################
# TODO: visualize activity in PO7/PO8 for each epoch
# EGI to 10-20 mapping: https://www.egi.com/images/HydroCelGSN_10-10.pdf

# TODO: calculate difference of PO7-PO8 activity for each epoch
#  check if N2pc exists and if it precedes the saccade
#  regress N2pc amplitude/latency with saccade amplitude and latency


# %%
##############################################
# Calc Evoked Activity in PO7/PO8
# EGI to 10-20 mapping: https://www.egi.com/images/HydroCelGSN_10-10.pdf

# TODO - actually we don't need this code cell - delete it


# REFERENCE = 'average'
REFERENCE = 'Cz'
on_epochs.set_eeg_reference(ref_channels=[REFERENCE])

evoked = on_epochs.average(picks=['E65', 'E90', 'rEOG'], by_event_type=False)   # EGI-128 equivalent to PO7/PO8
evoked.comment = 'Stimulut Onset'

evoked.plot(scalings=VISUALIZATION_SCALING)

evoked_fig = mne.viz.plot_compare_evokeds(
    evoked, picks='all',
    title="Evoked Response Comparison",
    # block=True,
)
evoked_fig.show()

# %%
##############################################
# Save the MNE-raw struct to file

split_path = PATH.split('\\')
new_dir, subj_id, basename = split_path[:-2], split_path[-2], split_path[-1].split('.')[0]
new_dir = os.path.join(*new_dir, 'mne_raw', subj_id)
os.makedirs(new_dir, exist_ok=True)
PATH_RAW = os.path.join(new_dir, f'{basename}.fif.gz')

raw.save(PATH_RAW, overwrite=True)
new_raw = mne.io.read_raw_fif(PATH_RAW, preload=True)


#########################################
# Contra-lateral Difference, following the analysis performed by Talcott et al., 2023 (https://doi.org/10.3758/s13414-023-02775-5)
# TODO: artifact/blink rejection or ICA - use synch_min or synch_max?
