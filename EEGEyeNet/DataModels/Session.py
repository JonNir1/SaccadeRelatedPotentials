import os.path
from abc import ABC, abstractmethod
from typing import final, Dict
from enum import StrEnum, IntEnum

import numpy as np
import pandas as pd
from pymatreader import read_mat

from utils.array_utils import to_vector
from utils.calc_utils import calculate_sampling_rate


class SessionTaskType(StrEnum):
    ANTI_SACCADE = "anti_saccade"
    DOTS = "dots"
    VISUAL_SEARCH = "visual_search"


class DotsTaskBlockType(IntEnum):
    OUT_OF_BLOCK = 0
    BASIC = 1
    REVERSED = 2
    MIRRORED = 3
    REVERSED_MIRRORED = 4
    BASIC2 = 5


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
    def get_data(self) -> np.ndarray:
        return self._data

    @final
    def get_timestamps(self) -> np.ndarray:
        return self._timestamps

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



class DotsSession(BaseSession):
    _TASK_TYPE = SessionTaskType.DOTS
    __GRID_PREFIX_STR: str = "grid_"

    def __init__(
            self,
            subject: str,
            data: np.ndarray,
            timestamps: np.ndarray,
            events: pd.DataFrame,
            channel_locations: pd.DataFrame,
            session_num: int,
            reference: str = "average"
    ):
        super().__init__(subject, data, timestamps, events, channel_locations, reference)
        self._session_num = session_num

    @staticmethod
    def from_mat_file(path: str) -> "DotsSession":
        mat = read_mat(path)['sEEG']

        # extract metadata from path
        basename = os.path.basename(path)   # example: EP12_DOTS5_EEG.mat
        subject_id = basename.split("_")[0].capitalize()
        session_num = int(basename.split("_")[1][-1])

        # load metadata from mat file
        num_channels = mat['nbchan']
        num_samples = mat['pnts']
        sampling_rate = mat['srate']
        xmin, xmax = mat['xmin'], mat['xmax']   # not sure what these are for
        ref = mat['ref'].strip().lower()
        ref = "average" if ref == "averef" else ref

        # load timestamps and verify inputs
        timestamps = to_vector(mat['times'])    # timestamps in milliseconds: 1 x num_samples
        if timestamps.shape[0] != num_samples:
            raise AssertionError(f"Number of samples in timestamps ({timestamps.shape[0]}) must match metadata ({num_samples})")
        _sr = calculate_sampling_rate(timestamps, decimals=3)
        if not np.isclose(_sr, sampling_rate):
            raise AssertionError(f"Sampling rate calculated from timestamps ({_sr}) must match metadata ({sampling_rate})")

        # load channel data and verify inputs
        data = mat['data']  # channel data: num_channels x num_samples
        if data.shape[0] != num_channels:
            raise AssertionError(f"Number of channels in data ({data.shape[0]}) must match metadata ({num_channels})")
        if data.shape[1] != num_samples:
            raise AssertionError(f"Number of samples in data ({data.shape[1]}) must match metadata ({num_samples})")

        # clean gaze data - if X, Y, and Pupil are all 0, replace with NaN
        is_missing_gaze_data = np.all(data[130:] <= 0, axis=0)
        data[130:, is_missing_gaze_data] = np.nan

        # load events (triggers & ET events) into DataFrames
        events = DotsSession._events_from_dict(mat['event'])
        _grid_num = int((events['type'][
            events['type'].map(lambda val: str(val).startswith(DotsSession.__GRID_PREFIX_STR))
        ].iloc[0])[-1])
        if _grid_num != session_num:
            raise AssertionError(f"Grid number in events ({_grid_num}) must match metadata ({session_num})")

        # load channel locations into DataFrames
        channel_locs = DotsSession._channel_locations_from_dict(mat['chanlocs'])
        if len(channel_locs.index) != num_channels:
            raise AssertionError(f"Number of channel locations ({len(channel_locs.index)}) must match metadata ({num_channels})")
        return DotsSession(subject_id, data, timestamps, events, channel_locs, session_num, ref)


    @property
    def session_num(self) -> int:
        return self._session_num

    @staticmethod
    def _events_from_dict(events: Dict[str, list]) -> pd.DataFrame:
        events_df = pd.DataFrame(events)
        missing_columns = set(DotsSession._EVENT_COLUMNS) - set(events_df.columns)
        if missing_columns:
            raise ValueError(f"Missing columns in events DataFrame: {missing_columns}")

        # fill missing values with NaN
        for col in events_df.columns:
            if col in ['type', 'latency']:
                # don't touch these columns
                continue
            events_df.loc[events_df[col] <= 0, col] = np.nan
        # parse the "type" column
        events_df['old_type'] = events_df['type']
        events_df['type'] = DotsSession.__parse_event_types(events_df['type'])
        # add "block" column: basic -> reversed -> mirrored -> reversed_mirrored -> basic2
        events_df['block'] = DotsSession.__extract_block_type(events_df['type'])
        return events_df

    @staticmethod
    def _channel_locations_from_dict(channel_locs: Dict[str, list]) -> pd.DataFrame:
        channel_locs_df = pd.DataFrame(channel_locs)
        missing_columns = set(DotsSession._CHANNEL_LOCATION_COLUMNS) - set(channel_locs_df.columns)
        if missing_columns:
            raise ValueError(f"Missing columns in channel locations DataFrame: {missing_columns}")
        channel_locs_df['labels'] = channel_locs_df['labels'].map(lambda val: val.strip())
        channel_locs_df.loc[channel_locs_df['labels'] == 'TIME', 'labels'] = 'ET_TIME'

        # parse channel types to MNE channel types
        channel_locs_df.loc[channel_locs_df['labels'].map(lambda lbl: "ET_TIME" in lbl.upper()), 'type'] = 'eyegaze'
        channel_locs_df.loc[channel_locs_df['labels'].map(lambda lbl: "GAZE" in lbl.upper()), 'type'] = 'eyegaze'
        channel_locs_df.loc[channel_locs_df['labels'].map(lambda lbl: "AREA" in lbl.upper()), 'type'] = 'pupil'
        is_eog = channel_locs_df['labels'].map(
            lambda lbl: lbl.upper() in [f"E{i}" for i in range(125, 129)]  # E125-E128 are EOG channels     # TODO: verify this is correct
        )
        channel_locs_df.loc[is_eog, 'type'] = 'eog'
        channel_locs_df['type'] = channel_locs_df['type'].map(
            lambda val: str(val).strip().replace('[', '').replace(']', '')
        ).map(lambda val: val if val else 'eeg')

        return channel_locs_df

    @staticmethod
    def __parse_event_types(event_type: pd.Series) -> pd.Series:
        event_type = event_type.map(lambda val: str(val).strip()).map(lambda val: int(val) if val.isnumeric() else val)
        event_type = event_type.replace({41: "stim_off", 55: "block_on", 56: "block_off"})
        block_on_idxs = event_type[event_type == "block_on"].index

        # find grid number: the first event labelled 12-17 before the first block_on event
        events_preceding_blocks = event_type.iloc[:block_on_idxs.min()]
        is_gridnum_event = np.isin(events_preceding_blocks, np.arange(12, 18))
        if is_gridnum_event.sum() == 0:
            raise ValueError("No grid number event found before first `block_on` event")
        if is_gridnum_event.sum() > 1:
            raise ValueError("Multiple grid number events found before first `block_on` event")
        gridnum_idx = events_preceding_blocks[is_gridnum_event].index.min()
        grid_number = int(event_type[gridnum_idx]) - 11     # grids 1-6 are labelled 12-17
        event_type[gridnum_idx] = f"{DotsSession.__GRID_PREFIX_STR}{grid_number}"
        return event_type

    @staticmethod
    def __extract_block_type(event_type: pd.Series) -> pd.Series:
        block_on_idxs = event_type[(event_type == "block_on") | (event_type == 55)].index
        block_off_idxs = event_type[(event_type == "block_off") | (event_type == 56)].index
        if len(block_on_idxs) != len(block_off_idxs):
            raise ValueError("Number of block_on and block_off events must match")
        if len(block_on_idxs) != 5:
            raise ValueError(f"Number of blocks must be 5, found {len(block_on_idxs)}")
        block_type = pd.Series(DotsTaskBlockType.OUT_OF_BLOCK, index=event_type.index)
        for i, (on, off) in enumerate(zip(block_on_idxs, block_off_idxs)):
            block_type.loc[on:off] = DotsTaskBlockType(i + 1)
        return block_type

    def __repr__(self):
        return f"{super().__repr__()}{self.session_num}"
