import os.path
from abc import ABC, abstractmethod
from typing import final, Dict
from enum import StrEnum, IntEnum

import numpy as np
import pandas as pd
from mne.io import Raw

from utils.array_utils import to_vector
from utils.calc_utils import calculate_sampling_rate


class SessionTaskType(StrEnum):
    ANTI_SACCADE = "anti_saccade"
    DOTS = "dots"
    VISUAL_SEARCH = "visual_search"


class EyeMovementType(IntEnum):
    UNDEFINED = 0
    FIXATION = 1
    SACCADE = 2
    BLINK = 5


class BaseSession(ABC):

    _TASK_TYPE: SessionTaskType
    _EVENT_COLUMNS = [
        'type', 'latency', 'duration', 'endtime',
        'sac_amplitude', 'sac_endpos_x', 'sac_endpos_y',
        'sac_startpos_x', 'sac_startpos_y', 'sac_vmax',
        'fix_avgpos_x', 'fix_avgpos_y', 'fix_avgpupilsize'
    ]
    _CHANNEL_LOCATION_COLUMNS = [
        'labels', 'Y', 'X', 'Z',
        'sph_theta', 'sph_phi', 'sph_radius', 'theta', 'radius',
    ]

    def __init__(
            self,
            subject: str,
            data: np.ndarray,
            timestamps: np.ndarray,
            events: pd.DataFrame,
            channel_locations: pd.DataFrame,
            reference: str = "average"
    ):
        self._subject = subject.strip().upper()
        self._data = data
        self._timestamps = timestamps
        self._reference = reference.strip().lower()
        self._sr = calculate_sampling_rate(timestamps)

        # Events/Triggers
        self._verify_events_input(events)
        self._events = events

        # Channel Locations
        self._verify_channel_locations_input(channel_locations)
        self._channel_locations = channel_locations

    @staticmethod
    @abstractmethod
    def from_mat_file(path: str) -> "BaseSession":
        raise NotImplementedError

    @abstractmethod
    def to_mne(self) -> Raw:
        raise NotImplementedError

    @final
    @property
    def subject(self) -> str:
        return self._subject

    @final
    @property
    def num_channels(self) -> int:
        return self._data.shape[0]

    @final
    @property
    def num_samples(self) -> int:
        return self._data.shape[1]

    @final
    @property
    def sampling_rate(self) -> float:
        return self._sr

    @final
    @property
    def reference(self) -> str:
        return self._reference

    @final
    @property
    def task_type(self) -> SessionTaskType:
        return self.__class__._TASK_TYPE

    @final
    def get_timestamps(self) -> np.ndarray:
        return self._timestamps

    @final
    def get_data(self) -> np.ndarray:
        return self._data

    @final
    def get_gaze_data(self) -> pd.DataFrame:
        # extract gaze data
        ts = self.get_timestamps()
        gaze = self._data[130:]
        gaze = pd.DataFrame(
            np.vstack((ts, gaze)).T, columns=['t', 'x', 'y', 'pupil']
        ).sort_values('t').reset_index(drop=True)

        # add label column for eye movement type
        gaze['label'] = EyeMovementType.UNDEFINED
        events = self.get_events()
        is_str_event = events['type'].map(lambda x: isinstance(x, str))
        for em in EyeMovementType:
            if em == EyeMovementType.UNDEFINED:
                continue
            is_em = events.loc[is_str_event, 'type'].map(lambda evnt: em.name.lower() in evnt.lower())
            is_em_sample = np.any(
                (gaze.index.to_numpy() >= events.loc[is_em.index[is_em], 'latency'].to_numpy()[:, None]) &
                (gaze.index.to_numpy() <= events.loc[is_em.index[is_em], 'endtime'].to_numpy()[:, None]),
                axis=0
            )
            gaze.loc[is_em_sample, 'label'] = em
        return gaze

    @final
    def get_events(self) -> pd.DataFrame:
        return self._events

    @final
    def get_channel_locations(self) -> pd.DataFrame:
        return self._channel_locations

    @final
    def get_channel_labels(self) -> np.ndarray:
        chan_locs = self.get_channel_locations()
        return chan_locs.labels

    @final
    def _verify_events_input(self, events: pd.DataFrame):
        missing_columns = set(self._EVENT_COLUMNS) - set(events.columns)
        if missing_columns:
            raise ValueError(f"Missing columns in events DataFrame: {missing_columns}")

    @final
    def _verify_channel_locations_input(self, channel_locations: pd.DataFrame):
        missing_columns = set(self._CHANNEL_LOCATION_COLUMNS) - set(channel_locations.columns)
        if missing_columns:
            raise ValueError(f"Missing columns in channel_locations DataFrame: {missing_columns}")
        if not channel_locations.labels.is_unique:
            raise ValueError("Channel labels must be unique")
        if not channel_locations.shape[0] == self.num_channels:
            raise ValueError(
                f"Number of channel locations ({channel_locations.shape[0]}) must match number of channels in data ({self.num_channels})"
            )

    def __eq__(self, other):
        if not isinstance(other, BaseSession):
            return False
        if self.subject != other.subject:
            return False
        if self.task_type != other.task_type:
            return False
        if self.reference != other.reference:
            return False
        if not np.equal(self.get_data(), other.get_data()).all():
            return False
        if not np.equal(self.get_timestamps(), other.get_timestamps()).all():
            return False
        if not self.get_events().equals(other.get_events()):
            return False
        if not self.get_channel_locations().equals(other.get_channel_locations()):
            return False
        return True

    @final
    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return f"{self.subject}_{self.__class__.__name__}"

    @final
    def __str__(self):
        return self.__repr__()

