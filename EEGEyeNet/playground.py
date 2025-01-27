import os
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import pickle as pkl
import mne

from pymatreader import read_mat

from EEGEyeNet.DataModels.Session import DotsSession


PATH = r'C:\Users\jonathanni\Desktop\EEGEyeNet\dots_data\synchronised_min\EP12\EP12_DOTS5_EEG.mat'
# PATH = r'C:\Users\nirjo\Desktop\SRP\data\EEGEyeNet\dots_data\sunchronised_min\EP12\EP12_DOTS1_EEG.mat'


ses = DotsSession.from_mat_file(PATH)