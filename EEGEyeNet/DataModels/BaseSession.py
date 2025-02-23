from abc import ABC, abstractmethod
from typing import final, Dict, Union, Optional
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
    def to_mne(self) -> (Raw, Dict[int, Union[int, str]]):
        """
        Convert the session data to an MNE Raw object, and return it along with a dictionary mapping event trigger
        values to event names.
        :return:
            raw: MNE Raw object
            event_dict: Dictionary mapping event trigger values to event names
        """
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
    def get_channel(self, channel: Union[str, int]) -> np.ndarray:
        """
        Get the data for a specific channel/electrode in the EGI-128 system. Central channels (Fz, Cz, Pz, Oz) can be
        specified by their label, while other channels can be specified by their electrode number/name (for example,
        electrode 62 can be specified as `62`, `E62`, or `Pz`).
        :param channel: The channel label or electrode number.
        :return: The data for the specified channel, with shape (N,) where N is the number of samples in the session.
        """
        central_channels = {'Fz': 'E11', 'Cz': 'Cz', 'Pz': 'E62', 'Oz': 'E75'}
        channel = central_channels.get(channel.capitalize(), channel)
        if isinstance(channel, int):
            channel = f"E{channel}"
        if isinstance(channel, str):
            channel = channel.upper().strip()
        else:
            raise TypeError(f"Invalid channel. Should be a string or integer, not {type(channel)}")
        return self._data[self.get_channel_locations().labels == channel].flatten()

    @final
    def get_eog(self) -> np.ndarray:
        """ Returns channels where the `type` (in the channel locations table) is 'eog'. """
        data = self.get_data()
        channel_locations = self.get_channel_locations()
        eog_data = data[channel_locations['type'] == 'eog']
        return eog_data

    @final
    def get_gaze_data(self) -> pd.DataFrame:
        """
        Extracts the gaze data from relevant channels (X: 130, Y: 131, Pupil: 132), along with the EEG timestamps
        (discarding the ET timestamps). Also computes the sample-by-sample label based on the ET events.

        :return: a pd.DataFrame of shape 5×N, where N is the number of samples in the session, and rows are `t`, `x`,
            `y`, `pupil`, and `label`.
        """
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
            is_em_idx = np.any(
                (gaze.index.to_numpy() >= events.loc[is_em.index[is_em], 'latency'].to_numpy()[:, None]) &
                (gaze.index.to_numpy() <= events.loc[is_em.index[is_em], 'endtime'].to_numpy()[:, None]),
                axis=0
            )
            gaze.loc[is_em_idx, 'label'] = em
        return gaze.T

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
    def calculate_radial_eog(self, ref: Union[str, int]) -> np.ndarray:
        """
        Calculates the "radial" EOG signal, using the method introduced by Keren, Yuval-Grinberg & Deouell, 2010 (https://doi.org/10.1016/j.neuroimage.2009.10.057):
        The rEOG is calculated by taking the mean of all EOG electrodes and subtracting the reference electrode,
        which should be a central electrode (Cz, Pz, or Oz).

        :return np.ndarray: The radial EOG signal, shape (N,) where N is the number of samples in the session.
        :raises ValueError: If the reference electrode is not one of 'Cz', 'Pz' (E62), or 'Oz' (E75).
        """
        eog_data = self.get_eog()
        if ref.lower() in {'cz'}:
            ref_data = self.get_channel('Cz')
        elif ref.lower() in {'pz', 'e62', 62}:
            ref_data = self.get_channel('E62')
        elif ref.lower() in {'oz', 'e75', 75}:
            ref_data = self.get_channel('E75')
        else:
            raise ValueError(f"Invalid rEOG ref: {ref}. Must be a central electrode: 'Cz', 'Pz', or 'Oz'.")
        return np.mean(eog_data, axis=0) - ref_data

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

    @staticmethod
    @final
    def _parse_raw_channel_locations(channel_locs: Dict[str, list]) -> pd.DataFrame:
        """
        Extracts the `channel_locations` table from the raw `channel_locs` struct in the sEEG mat file, using the info
        provided in the "data structure" appendix to EEGEyeNet's documentation: https://osf.io/ktv7m/
        """
        channel_locs_df = pd.DataFrame(channel_locs)
        missing_columns = set(BaseSession._CHANNEL_LOCATION_COLUMNS) - set(channel_locs_df.columns)
        if missing_columns:
            raise ValueError(f"Missing columns in channel locations DataFrame: {missing_columns}")
        channel_locs_df['labels'] = channel_locs_df['labels'].map(lambda val: val.strip())
        channel_locs_df.loc[channel_locs_df['labels'] == 'TIME', 'labels'] = 'ET_TIME'

        # parse channel types to MNE channel types
        channel_locs_df.loc[channel_locs_df['labels'].map(lambda lbl: "ET_TIME" in lbl.upper()), 'type'] = 'eyegaze'
        channel_locs_df.loc[channel_locs_df['labels'].map(lambda lbl: "GAZE" in lbl.upper()), 'type'] = 'eyegaze'
        channel_locs_df.loc[channel_locs_df['labels'].map(lambda lbl: "AREA" in lbl.upper()), 'type'] = 'pupil'
        channel_locs_df.loc[
            # EOG channels are based on Jia & Tyler, 2019 (https://doi.org/10.3758/s13428-019-01280-8), Methods section "Eye Tracking"
            channel_locs_df['labels'].map(lambda lbl: lbl.upper() in ['E25', 'E127', 'E8', 'E126', 'E32', 'E1', 'E17']),
            'type'] = 'eog'
        channel_locs_df['type'] = channel_locs_df['type'].map(
            # fill cells with no `type` value with the type 'eeg'
            lambda val: str(val).strip().lower().replace('[', '').replace(']', '')
        ).map(lambda val: val if val else 'eeg')

        # replace empty array cells with NaN
        channel_locs_df = channel_locs_df.map(
            lambda val: np.nan if isinstance(val, np.ndarray) and len(val) == 0 else val
        )
        return channel_locs_df

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
