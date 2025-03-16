import os

import numpy as np
import pandas as pd
import mne

import matplotlib
matplotlib.use('TkAgg')
# mne.viz.set_browser_backend('qt')

from EEGEyeNet.DataModels.DotsSession import DotsSession

_SUBJ_ID = "EP12"
_BASE_PATH = r'C:\Users\jonathanni\Desktop\EEGEyeNet\dots_data\synchronised_min'  # lab
# _BASE_PATH = r'C:\Users\nirjo\Desktop\SRP\data\EEGEyeNet\dots_data\sunchronised_min'  # home
_FILE_PATH = f'{_SUBJ_ID}\{_SUBJ_ID}_DOTS3_EEG.mat'
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
#     block=True,
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
# Filter

import mne_scripts.helpers.raw_helpers as rh

NOTCH_FREQ = 50
LOW_FREQ, HIGH_FREQ = 0.1, 100

raw_unfiltered = raw.copy()     # keep a copy of the unfiltered data
raw = rh.apply_lowpass_filter(raw, HIGH_FREQ, include_eog=True, inplace=True, suppress_warnings=True)
raw = rh.apply_highpass_filter(raw, LOW_FREQ, include_eog=True, inplace=True, suppress_warnings=True)
raw = rh.apply_notch_filter(raw, NOTCH_FREQ, include_eog=True, inplace=True, suppress_warnings=True)


# psd = raw.copy().compute_psd(n_fft=512, verbose=False, exclude=[EEG_REF])
# psd.plot(picks=['eeg', 'eog'])

# %%
##############################################
# Annotate Blinks

import mne_scripts.helpers.annotation_helpers as annh

BEFORE_BLINK, AFTER_BLINK = 250, 250    # annotate 250ms before & after each detected blink
MERGE_BLINKS_MS = 25                    # merge blinks that are within 25ms of each other

raw.set_annotations(annh.blink_annotations(
    raw, 'STI_ET', {215, 216}, 'auto', BEFORE_BLINK, AFTER_BLINK, MERGE_BLINKS_MS
))


# raw.plot(
#     events=mne_events,
#     scalings=VISUALIZATION_SCALING, n_channels=10,
#     block=True,
# )


# %%
##############################################
# ICA

EPOCH_TMIN, EPOCH_TMAX, EPOCH_BASELINE = -0.5, 1.5, (-0.4, -0.05)

# extract data from blink epochs
blink_epochs = mne.Epochs(
    raw, events=None,       # specify no events to get Epochs from Annotations
    tmin=EPOCH_TMIN, tmax=EPOCH_TMAX, baseline=EPOCH_BASELINE,
    reject=None, reject_tmax=0.5, reject_tmin=-0.1,
    verbose=False,
)
blink_raw = mne.io.RawArray(np.hstack(blink_epochs.get_data(verbose=False).copy()), blink_epochs.info, verbose=False)

# extract data from trial epochs
trial_onset_epochs = mne.Epochs(
    raw,
    mne_events, event_id={k: v for k, v in event_dict.items() if k.startswith('stim') and not k.endswith('off')},
    tmin=EPOCH_TMIN, tmax=EPOCH_TMAX, baseline=EPOCH_BASELINE,
    verbose=False,
)
trial_raw = mne.io.RawArray(
    np.hstack(trial_onset_epochs.get_data(verbose=False).copy()), trial_onset_epochs.info, verbose=False
)

# concat blink and trial data
raw_for_ica = mne.concatenate_raws([trial_raw, blink_raw], verbose=False)
del trial_raw, blink_raw

# fit ICA
ica = mne.preprocessing.ICA(
    n_components=20, random_state=42, max_iter=800, method='picard', fit_params=dict(extended=True),
)
ica.fit(raw_for_ica, reject=dict(eeg=400e-6), reject_by_annotation=True, picks=["eeg", "eog"], verbose=True)

## TODO: START FROM HERE

# ica.plot_components(picks=range(20))
#
# # Gal's step 9-10
# ica.plot_sources(raw)
# ica.plot_properties(trial_onset_epochs, picks=ica.exclude, psd_args={'fmax': 40})
#
# # Gal's step 11
# ica_raw = ica.get_sources(raw)
# ch_dict = {name: "eeg" for name in ica_raw.ch_names}
# ica_raw.set_channel_types(ch_dict)
# ica_raw.info["bads"] = []
# ica_raw.compute_psd(
#     picks=list(np.array(ica_raw.ch_names)[ica.exclude]),
#     n_overlap=int(0.2 * raw.info['sfreq']),
#     n_fft=int(2 * raw.info['sfreq'])
# ).plot()

# # Gal's step 12: apply ica
# unfiltered_raw_unclean=unfiltered_raw.copy()
# unfiltered_raw = ica.apply(unfiltered_raw)
# unfiltered_raw.interpolate_bads()
# unfiltered_raw.plot(n_channels=40, duration=10, events=events_updated)  # make sure you don't see blinks
# exclusions = pd.DataFrame({"excluded": ica.exclude})
# exclusions.to_csv(join(save_dir, f"sub-{subject_num}{add}-{low_cutoff_freq:.2f}hpf-infomax_ex-ica-rejected.csv"))
# unfiltered_raw.save(join(save_dir, f"sub-{subject_num}{add}-unfiltered-clean-raw.fif"), overwrite=True)







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
