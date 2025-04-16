import os
import copy
from abc import ABC, abstractmethod
from typing import final, Optional, Dict, Union
from enum import StrEnum, IntEnum

import pandas as pd
from pymatreader import read_mat
from mne.io import Raw

from mne_scripts.helpers.utils import *


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

    _TASK_TYPE: SessionTaskType

    def __init__(
            self,
            subject: str,
            data: np.ndarray,
            timestamps: np.ndarray,
            events: pd.DataFrame,
            channel_locations: pd.DataFrame,
            reference: Optional[str] = None
    ):
        self._subject = subject.strip().upper()
        self._data = data
        self._timestamps = timestamps
        self._reference = None if reference is None else reference.strip().lower()
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
        # TODO: make this a Factory Method once other session classes are implemented
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
        # TODO: when additional session classes are implemented, this method should contain common logic with an _impl
        #  method that is overridden by the subclasses
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
    def reference(self) -> Optional[str]:
        return self._reference

    @final
    @reference.setter
    def reference(self, ref: Optional[str]):
        if self.reference == ref:
            return
        # TODO
        raise NotImplementedError("Re-setting the reference electrode is not supported yet")

    @final
    @property
    def task_type(self) -> SessionTaskType:
        return self.__class__._TASK_TYPE

    @classmethod
    @final
    def units(cls, key: str) -> str:
        """
        Returns the measurement units for the specified key (e.g., 'eeg', 'eog', 'eyegaze', 'pupil', 'time'), based on
        the units used in the EEGEyeNet dataset. If the key is not recognized, returns 'unknown'.
        """
        eegeyenet_units = dict(eeg='µV', eog='µV', eyegaze='pixels', pupil='AU', time='ms')
        return eegeyenet_units.get(key, 'unknown')

    @classmethod
    @final
    def para_ocular_electrodes(cls) -> list[str]:
        """
        Returns a list of electrode names for the para-ocular electrodes used in the EGI-128 system.
        We follow the convention used by Jia & Tyler, 2019 (https://doi.org/10.3758/s13428-019-01280-8), who extracted
        EOG data using channels E25, E127, E8, E126, E32, E1, and E17. We add channels E125 and E128 as well.
        """
        return ['E25', 'E127', 'E8', 'E126', 'E32', 'E1', 'E17', 'E125', 'E128']

    @final
    def get_timestamps(self) -> np.ndarray:
        return self._timestamps

    @final
    def get_channel_locations(self) -> pd.DataFrame:
        return self._channel_locations

    @final
    def get_channel_labels(self) -> np.ndarray:
        chan_locs = self.get_channel_locations()
        return chan_locs.labels

    @final
    def get_events(self) -> pd.DataFrame:
        return self._events

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
            channel = channel.strip().capitalize()
            if channel in central_channels.keys():
                channel = central_channels[channel]
            else:
                channel = channel.upper()
        else:
            raise TypeError(f"Invalid channel. Should be a string or integer, not {type(channel)}")
        return self._data[self.get_channel_locations().labels == channel].flatten()

    @final
    def get_data(self, as_frame: bool = True) -> Union[np.ndarray, pd.DataFrame]:
        data = copy.deepcopy(self._data)
        if not as_frame:
            return data
        data = pd.DataFrame(data, index=self.get_channel_labels())
        data.index.name = "channel"
        data.columns.name = "sample"
        return data

    @final
    def get_eeg(self, as_frame: bool = True) -> Union[np.ndarray, pd.DataFrame]:
        """ Returns channels where the `type` (in the channel locations table) is 'eeg'. """
        data = self.get_data(as_frame=False)
        channel_locations = self.get_channel_locations()
        eeg_data = data[channel_locations['type'] == 'eeg']
        if not as_frame:
            return eeg_data
        eeg_data = pd.DataFrame(eeg_data, index=channel_locations["labels"])
        eeg_data.index.name = "channel"
        eeg_data.columns.name = "sample"
        return eeg_data

    @final
    def get_eog(self, as_frame: bool = True) -> Union[np.ndarray, pd.DataFrame]:
        """ Returns channels where the `type` (in the channel locations table) is 'eog'. """
        data = self.get_data(as_frame=False)
        channel_locations = self.get_channel_locations()
        eog_data = data[channel_locations['type'] == 'eog']
        if not as_frame:
            return eog_data
        eog_data = pd.DataFrame(eog_data, index=channel_locations["labels"])
        eog_data.index.name = "channel"
        eog_data.columns.name = "sample"
        return eog_data

    @final
    def get_gaze(self, as_frame: bool = True) -> Union[np.ndarray, pd.DataFrame]:
        """
        Returns channels where the `type` (in the channel locations table) is 'eyegaze' or 'pupil'.
        If `as_frame` is True, appends the timestamps and sample labels to the gaze data, and returns a DataFrame.
        """
        data = self.get_data(as_frame=False)
        channel_locations = self.get_channel_locations()
        is_et = np.isin(channel_locations['type'], ['eyegaze', 'pupil'])
        gaze_data = data[is_et]
        if not as_frame:
            return gaze_data
        gaze_data = pd.DataFrame(gaze_data, index=channel_locations.loc[is_et, "labels"]).rename(index={
            'L-GAZE-X': 'x', 'L-GAZE-Y': 'y', 'L-AREA': 'pupil',
            'R-GAZE-X': 'x', 'R-GAZE-Y': 'y', 'R-AREA': 'pupil'
        }).T
        gaze_data.columns.name, gaze_data.index.name = "channel", "sample"
        gaze_data['t'] = self.get_timestamps()

        # add label column for eye movement type
        gaze_data['label'] = EyeMovementType.UNDEFINED
        events = self.get_events()
        is_str_event = events['type'].map(lambda x: isinstance(x, str))
        for em in EyeMovementType:
            if em == EyeMovementType.UNDEFINED:
                continue
            is_em = events.loc[is_str_event, 'type'].map(lambda evnt: em.name.lower() in evnt.lower())
            is_em_idx = np.any(
                (gaze_data.index.to_numpy() >= events.loc[is_em.index[is_em], 'latency'].to_numpy()[:, None]) &
                (gaze_data.index.to_numpy() <= events.loc[is_em.index[is_em], 'endtime'].to_numpy()[:, None]),
                axis=0
            )
            gaze_data.loc[is_em_idx, 'label'] = em
        return gaze_data.T

    @final
    def calculate_radial_eog(self, ref: Union[str, int]) -> np.ndarray:
        """
        Calculates the "radial" EOG signal, using the method introduced by Keren, Yuval-Grinberg & Deouell, 2010
        (https://doi.org/10.1016/j.neuroimage.2009.10.057). The radial EOG signal is calculated by taking the mean of
        electrodes around the eyes (para-ocular electrodes) and subtracting the reference electrode, which should be a
        central electrode (Cz, Pz, or Oz).
        NOTE: Jia & Tyler, 2019 (https://doi.org/10.3758/s13428-019-01280-8) extract EOG data using channels E25, E127,
        E8, E126, E32, E1, and E17 (EGI-128 system). We add channels E125 and E128 to this list.

        :return np.ndarray: The radial EOG signal, shape (N,) where N is the number of samples in the session.
        :raises ValueError: If the reference electrode is not one of 'Cz', 'Pz' (E62), or 'Oz' (E75).
        """
        para_ocular_electrodes = self.para_ocular_electrodes()
        para_ocular_data = np.vstack([self.get_channel(e) for e in para_ocular_electrodes])
        para_ocular_mean = np.mean(para_ocular_data, axis=0)
        if ref.lower() in {'cz'}:
            ref_data = self.get_channel('Cz')
        elif ref.lower() in {'pz', 'e62', 62}:
            ref_data = self.get_channel('E62')
        elif ref.lower() in {'oz', 'e75', 75}:
            ref_data = self.get_channel('E75')
        else:
            raise ValueError(f"Invalid rEOG ref: {ref}. Must be a central electrode: 'Cz', 'Pz', or 'Oz'.")
        return np.mean(para_ocular_mean, axis=0) - ref_data

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
    def _parse_mat_file(path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        mat = read_mat(path)['sEEG']
        # load metadata from mat file
        num_channels = mat['nbchan']
        num_samples = mat['pnts']
        sampling_rate = mat['srate']
        xmin, xmax = mat['xmin'], mat['xmax']       # not sure what these are for
        ref = mat['ref'].strip().lower()
        ref = None if ref == "averef" else ref      # 'averef' is the default reference in EEGLAB

        # load timestamps and verify inputs
        timestamps = to_vector(mat['times'])  # timestamps in milliseconds: 1 x num_samples
        if timestamps.shape[0] != num_samples:
            raise AssertionError(
                f"Number of samples in timestamps ({timestamps.shape[0]}) must match metadata ({num_samples})")
        _sr = calculate_sampling_rate(timestamps, decimals=3)
        if not np.isclose(_sr, sampling_rate):
            raise AssertionError(
                f"Sampling rate calculated from timestamps ({_sr}) must match metadata ({sampling_rate})")

        # load channel data and verify inputs
        data = mat['data']  # channel data: num_channels x num_samples
        if data.shape[0] != num_channels:
            raise AssertionError(f"Number of channels in data ({data.shape[0]}) must match metadata ({num_channels})")
        if data.shape[1] != num_samples:
            raise AssertionError(f"Number of samples in data ({data.shape[1]}) must match metadata ({num_samples})")

        # load channel locations into DataFrame and verify inputs
        channel_locs = BaseSession.__parse_raw_channel_locations(mat['chanlocs'])
        if len(channel_locs.index) != num_channels:
            raise AssertionError(
                f"Number of channel locations ({len(channel_locs.index)}) must match metadata ({num_channels})")

        # load events into DataFrame
        events = BaseSession.__parse_raw_events(mat['event'])

        # clean gaze data - if X, Y, and Pupil are all 0, replace with NaN
        is_gaze_channel = np.isin(channel_locs['labels'], ['L-GAZE-X', 'L-GAZE-Y', 'R-GAZE-X', 'R-GAZE-Y'])
        is_missing_gaze = np.all(data[is_gaze_channel, :] <= 0, axis=0)
        is_pupil_channel = np.isin(channel_locs['labels'], ['L-AREA', 'R-AREA'])
        is_missing_pupil = (data[is_pupil_channel] <= 0).flatten()
        data[is_gaze_channel | is_pupil_channel][:, is_missing_gaze | is_missing_pupil] = np.nan

        return data, timestamps, events, channel_locs, ref

    @staticmethod
    def __parse_raw_channel_locations(channel_locs: Dict[str, list]) -> pd.DataFrame:
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
        channel_locs_df.loc[channel_locs_df['labels'].map(lambda lbl: "ET_TIME" in lbl.upper()), 'type'] = 'misc'
        channel_locs_df.loc[channel_locs_df['labels'].map(lambda lbl: "GAZE" in lbl.upper()), 'type'] = 'eyegaze'
        channel_locs_df.loc[channel_locs_df['labels'].map(lambda lbl: "AREA" in lbl.upper()), 'type'] = 'pupil'
        channel_locs_df['type'] = channel_locs_df['type'].map(
            # fill cells with no `type` value with the type 'eeg'
            lambda val: str(val).strip().lower().replace('[', '').replace(']', '')
        ).map(lambda val: val if val else 'eeg')

        # replace empty array cells with NaN
        channel_locs_df = channel_locs_df.map(
            lambda val: np.nan if isinstance(val, np.ndarray) and len(val) == 0 else val
        )
        return channel_locs_df

    @staticmethod
    @final
    def __parse_raw_events(events: Dict[str, list]) -> pd.DataFrame:
        events_df = pd.DataFrame(events)
        missing_columns = set(BaseSession._EVENT_COLUMNS) - set(events_df.columns)
        if missing_columns:
            raise ValueError(f"Missing columns in events DataFrame: {missing_columns}")

        # calculate end time for each event (including 0-duration events)
        new_endtime = events_df['latency'] + events_df['duration']
        is_zero_dur = events_df['duration'] == 0
        new_endtime[~is_zero_dur] -= 1  # start count from zero so subtract 1 from end time
        if not np.all(new_endtime[is_zero_dur] == events_df['latency'][is_zero_dur]):
            raise ValueError("Error in calculating end time for zero-duration events")
        orig_endtime = events_df['endtime']
        if not np.all((orig_endtime[~is_zero_dur] == new_endtime[~is_zero_dur])):
            raise ValueError("Error in calculating end time for non-zero events")
        events_df['endtime'] = new_endtime

        # fill missing values with NaN
        for col in events_df.columns:
            if col in ['type', 'latency', 'duration', 'endtime']:
                # don't touch these columns
                continue
            events_df.loc[events_df[col] <= 0, col] = np.nan
        return events_df

    def __eq__(self, other):
        if not isinstance(other, BaseSession):
            return False
        if self.subject != other.subject:
            return False
        if self.task_type != other.task_type:
            return False
        if self.reference != other.reference:
            return False
        if not np.equal(self.get_data(as_frame=False), other.get_data(as_frame=False)).all():
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
