import os

import plotly.express as px

###################
#     STRINGS     #
###################

DATA_STR = "data"
RESULTS_STR = "results"
FIGURES_STR = "Figures"
EPOCHS_STR = "Epochs"
SAMPLES_STR = "Samples"
CHANNELS_STR = "Channels"
EVENTS_STR = "Events"

####################
#  OTHER CONSTANTS #
####################

SAMPLING_FREQUENCY = 1024  # eeg sampling frequency
DEFAULT_COLORMAP = px.colors.qualitative.Dark24

###################
#     PATHS       #
###################

# _BASE_DIR = r'C:\Users\nirjo\Desktop\'
_BASE_DIR = r"Z:\Lab-Shared\Experiments\SaccadeRelatedPotentials"
DATA_DIR = os.path.join(_BASE_DIR, DATA_STR)
RESULTS_DIR = os.path.join(_BASE_DIR, RESULTS_STR)

