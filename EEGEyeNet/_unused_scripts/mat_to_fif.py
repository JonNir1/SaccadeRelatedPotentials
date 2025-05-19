import os
import pickle

from EEGEyeNet.DataModels.Dots import DotsSession

_BASE_PATH = r''      # path-to-EEGEyeNet-dots_data-dir
_BASE_PATH = r'C:\Users\nirjo\Desktop\SRP\data\EEGEyeNet\dots_data\synchronised_min'
_SUBJ_ID = "EP10"

session = DotsSession.from_mat(os.path.join(_BASE_PATH, _SUBJ_ID))
raw, event_dict = session.to_mne()

raw.save(os.path.join(_BASE_PATH, _SUBJ_ID, "concatenated_raw.fif"))
with open(os.path.join(_BASE_PATH, _SUBJ_ID, "event_dict.pkl", 'wb')) as evnts_file:
    pickle.dump(event_dict, evnts_file, protocol=pickle.HIGHEST_PROTOCOL)
