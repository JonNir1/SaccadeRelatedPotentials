import os
import pickle
from typing import Tuple, Dict

import mne
from tqdm import tqdm
from EEGEyeNet.DataModels.Dots import DotsSession

__RAW_FILE_NAME = "concatenated_raw.fif"
__EVENTS_FILE_NAME = "events_dict.pkl"


# %%
#########################

def concat_single_subject(subj_dir: str) -> Tuple[mne.io.Raw, Dict[str, int]]:
    """ Load the Dots session data for a single subject in MNE format and save it to disk if not already done. """
    if not os.path.exists(subj_dir) or not os.path.isdir(subj_dir):
        raise NotADirectoryError(subj_dir, "Subject directory does not exist")
    concatenated_path = os.path.join(subj_dir, __RAW_FILE_NAME)
    events_dict_path = os.path.join(subj_dir, __EVENTS_FILE_NAME)
    try:
        concatenated = mne.io.read_raw_fif(concatenated_path, verbose=False)
        with open(events_dict_path, 'rb') as evnts_file:
            event_dict = pickle.load(evnts_file)
        return concatenated, event_dict
    except FileNotFoundError:
        session = DotsSession.from_mat_files(subj_dir, verbose=False)
        concatenated, event_dict = session.to_mne()
        del session
        concatenated.save(concatenated_path, picks="all", verbose=False)
        with open(events_dict_path, 'wb') as evnts_file:
            pickle.dump(event_dict, evnts_file, protocol=pickle.HIGHEST_PROTOCOL)
        return concatenated, event_dict


def concat_all_sessions(base_dir: str):
    if not os.path.exists(base_dir):
        raise NotADirectoryError(base_dir, "Base directory does not exist")
    for subj in tqdm(os.listdir(base_dir), unit="session"):
        subj_dir = os.path.join(base_dir, subj)
        try:
            concat_single_subject(subj_dir)
        except NotADirectoryError as e:
            print(e)
        except FileNotFoundError as e:
            print(e)
        # except Exception as e:
        #     print(f"Error processing {subj}: {e}")
    print("Done!")

# %%
#########################


# _BASE_PATH = r'C:\Users\jonathanni\Desktop\EEGEyeNet\dots_data\synchronised_max'  # lab
_BASE_PATH = r'C:\Users\nirjo\Desktop\SRP\data\EEGEyeNet\dots_data\synchronised_max'  # home

concat_all_sessions(_BASE_PATH)
