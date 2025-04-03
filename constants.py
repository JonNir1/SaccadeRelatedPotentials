import os

import plotly.express as px

###################
#     STRINGS     #
###################

DATA_STR = "data"
RESULTS_STR = "results"
FIGURES_STR = "figures"
EPOCH_STR, EPOCHS_STR = "epoch", "epochs"
SAMPLE_STR, SAMPLES_STR = "sample", "samples"
CHANNEL_STR, CHANNELS_STR = "channel", "channels"
EVENT_STR, EVENTS_STR = "event", "events"
DURATION_STR = "duration"
AMPLITUDE_STR = "amplitude"
SACCADE_STR = "saccade"
SUBJECT_STR = "subject"
ONSET_STR = "onset"
OFFSET_STR = "offset"

####################
#  OTHER CONSTANTS #
####################

SAMPLING_FREQUENCY = 1024  # eeg sampling frequency
DEFAULT_COLORMAP = px.colors.qualitative.Dark24

###################
#     PATHS       #
###################

# _BASE_DIR = r'C:\Users\nirjo\Desktop\SRP'
_BASE_DIR = r"S:\Lab-Shared\Experiments\SaccadeRelatedPotentials"
DATA_DIR = os.path.join(_BASE_DIR, DATA_STR)
RESULTS_DIR = os.path.join(_BASE_DIR, RESULTS_STR)

