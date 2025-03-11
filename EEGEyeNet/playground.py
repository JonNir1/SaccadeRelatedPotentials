import os

import numpy as np
import pandas as pd
import mne

import matplotlib
matplotlib.use('Qt5Agg')

from EEGEyeNet.DataModels.DotsSession import DotsSession

_BASE_PATH = r'C:\Users\jonathanni\Desktop\EEGEyeNet\dots_data\synchronised_min'  # lab
# _BASE_PATH = r'C:\Users\nirjo\Desktop\SRP\data\EEGEyeNet\dots_data\sunchronised_min'  # home
_FILE_PATH = r'EP12\EP12_DOTS3_EEG.mat'
FULL_PATH = os.path.join(_BASE_PATH, _FILE_PATH)

EEG_REF = 'Cz'
VISUALIZATION_SCALING = dict(eeg=1e-4, eog=1e-4, eyegaze=5e2, pupil=5e2)

# %%
##############################################
# Load the data and convert to MNE format

ses = DotsSession.from_mat_file(FULL_PATH)
raw_unreferenced, event_dict = ses.to_mne()

stim_channel_names = pd.Series(raw_unreferenced.info['ch_names']).loc[pd.Series(raw_unreferenced.get_channel_types()) == 'stim'].tolist()
mne_events = mne.find_events(
    raw_unreferenced,
    stim_channel=stim_channel_names,
    output='onset',
    shortest_event=1,
    consecutive=True,
    verbose=False
)

# raw_unreferenced.plot(
#     # block=True,
#     scalings=VISUALIZATION_SCALING, n_channels=20,
#     # events=mne_events
# )

# %%
##############################################
# Re-Reference to Cz and Mark EOG channels

raw = raw_unreferenced.copy().set_eeg_reference(ref_channels=[EEG_REF], verbose=False)
raw.set_channel_types({
    # set electrodes around the eyes as EOG (see Jia & Tyler (2019): https://doi.org/10.3758/s13428-019-01280-8)
    ch: 'eog' for ch in ses.para_ocular_electrodes()
})

# raw.plot(
#     # block=True,
#     scalings=VISUALIZATION_SCALING, n_channels=20,
#     # events=mne_events
# )

# %%
##############################################
# Re-Reference and Filter

NOTCH_FREQ = 50
LOW_FREQ, HIGH_FREQ = 0.1, 100

# TODO: check if we need to specify `fir_design` and `picks`

raw_unfiltered = raw.copy()     # keep a copy of the unfiltered data
raw.filter(         # band-pass filter
    l_freq=LOW_FREQ, h_freq=HIGH_FREQ,
    fir_design='firwin',
    picks=["eeg", "eog"]
)
raw.notch_filter(   # remove AC line noise
    freqs=np.arange(NOTCH_FREQ, 1 + 5 * NOTCH_FREQ, NOTCH_FREQ),
    fir_design='firwin',
    picks=["eeg", "eog"]
)


# %%
##############################################
# Detect and Annotate Blinks

BEFORE_BLINK, AFTER_BLINK = 25, 25  # annotate 25ms before & after each detected blink

import mne_helpers.ica as mne_ica

blink_annots_et = mne_ica._eyetracking_blink_annotation(raw, 'STI_ET', {215, 216}, BEFORE_BLINK, AFTER_BLINK)
blink_annots_eog = mne_ica._eog_blink_annotation(raw, BEFORE_BLINK, AFTER_BLINK)
raw.set_annotations(blink_annots_et + blink_annots_eog)

# raw.plot(
#     events=mne_events,
#     scalings=VISUALIZATION_SCALING, n_channels=10,
#     block=True,
# )

# %%
##############################################
# Epoch trials based on `stim/{%d}` events

BEFORE_EPOCH_SEC, AFTER_EPOCH_SEC = 0.5, 1.5

# mne.viz.plot_events(
#     mne_events, sfreq=raw.info['sfreq'], event_id=event_dict,
#     first_samp=raw.first_samp, on_missing='ignore',
# )

epochs = mne.Epochs(
    raw, mne_events, event_id=event_dict,
    tmin=-1 * BEFORE_EPOCH_SEC, tmax=AFTER_EPOCH_SEC,
    reject_by_annotation=True, preload=True, on_missing='ignore'
)

stim_onset_epochs = epochs[[
    key for key in epochs.event_id.keys() if key.startswith('stim') and not key.endswith('off')
]]

# %%
##############################################
# TODO: visualize activity in PO7 (E65) and PO8 (E90) for each epoch
# EGI to 10-20 mapping: https://www.egi.com/images/HydroCelGSN_10-10.pdf

eog_channel_names = pd.Series(raw.info['ch_names']).loc[pd.Series(raw.get_channel_types()) == 'eog'].tolist()
gaze_channel_names = pd.Series(raw.info['ch_names']).loc[pd.Series(raw.get_channel_types()) == 'eyegaze'].tolist()
stim_onset_epochs_fig = stim_onset_epochs.plot(
    n_epochs=5, n_channels=20, events=True, scalings=VISUALIZATION_SCALING,
    picks=['E65', 'E90'] + eog_channel_names + gaze_channel_names,
    block=True,
)

# TODO: calculate difference of PO7-PO8 activity for each epoch
#  check if N2pc exists and if it precedes the saccade
#  regress N2pc amplitude/latency with saccade amplitude and latency

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
