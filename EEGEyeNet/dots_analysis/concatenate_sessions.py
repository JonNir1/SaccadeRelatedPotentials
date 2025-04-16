import os
import pickle
from typing import Tuple, Dict

import mne
from tqdm import tqdm
from EEGEyeNet.DataModels.DotsSession import DotsSession

_MAT_FILE_FORMAT = "%s_DOTS%d_EEG.mat"


# %%
#########################

def concatenate_single_subject(subj_dir: str) -> Tuple[mne.io.Raw, Dict[str, int]]:
    """
    Concatenate all sessions of a subject, saves the concatenated raw and event dictionary to disk, and returns them.
    """
    if not os.path.exists(subj_dir):
        raise NotADirectoryError(subj_dir, "Subject directory does not exist")
    concatenated_path = os.path.join(subj_dir, "concatenated_raw.fif")
    events_dict_path = os.path.join(subj_dir, "events_dict.pkl")
    try:
        concatenated = mne.io.read_raw_fif(concatenated_path, verbose=False)
        with open(events_dict_path, 'rb') as evnts_file:
            event_dict = pickle.load(evnts_file)
        return concatenated, event_dict
    except FileNotFoundError:
        subject_id = os.path.basename(subj_dir)     # path format: C:/path/to/data/SUBJECT-ID/
        raws = []
        for i in range(1, 7):
            mat_path = os.path.join(subj_dir, _MAT_FILE_FORMAT % (subject_id, i))
            try:
                raw, event_dict = DotsSession.from_mat_file(mat_path).to_mne()
                raws.append(raw)
                del raw, mat_path
            except FileNotFoundError:
                print(f"\nFile not found: {mat_path}")
                continue
        concatenated = mne.concatenate_raws(raws, verbose=False)
        del raws
        concatenated.save(concatenated_path, picks="all", verbose=False)
        with open(events_dict_path, 'wb') as evnts_file:
            pickle.dump(event_dict, evnts_file, protocol=pickle.HIGHEST_PROTOCOL)
        return concatenated, event_dict


def concat_all_subjects(base_dir: str):
    if not os.path.exists(base_dir):
        raise NotADirectoryError(base_dir, "Base directory does not exist")
    for subj in tqdm(os.listdir(base_dir)):
        subj_dir = os.path.join(base_dir, subj)
        if os.path.isdir(subj_dir):
            try:
                concatenate_single_subject(subj_dir)
            except NotADirectoryError as e:
                print(e)
            except FileNotFoundError as e:
                print(e)
            except Exception as e:
                print(f"Error processing {subj}: {e}")
    print("Done!")

# %%
#########################


# _BASE_PATH = r'C:\Users\jonathanni\Desktop\EEGEyeNet\dots_data\synchronised_min'  # lab
_BASE_PATH = r'C:\Users\nirjo\Desktop\SRP\data\EEGEyeNet\dots_data\sunchronised_min'  # home

concat_all_subjects(_BASE_PATH)
