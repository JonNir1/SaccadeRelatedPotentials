import os
from typing import Dict
from enum import IntEnum

import numpy as np
import pandas as pd
import mne
from pymatreader import read_mat

from EEGEyeNet.DataModels.BaseSession import BaseSession, SessionTaskType
from utils.array_utils import to_vector
from utils.calc_utils import calculate_sampling_rate


class DotsBlockTaskType(IntEnum):
    OUT_OF_BLOCK = 0
    BASIC = 1
    REVERSED = 2
    MIRRORED = 3
    REVERSED_MIRRORED = 4
    BASIC2 = 5



class DotsSession(BaseSession):
    _TASK_TYPE = SessionTaskType.DOTS
    __GRID_PREFIX_STR: str = "grid_"
    __EVENTS_DICT = {
        "stim_off": 41,
        "block_on": 55,
        "block_off": 56,

        "grid_1": 201,
        "grid_2": 202,
        "grid_3": 203,
        "grid_4": 204,
        "grid_5": 205,

        "L_fixation": 211,
        "R_fixation": 212,
        "L_saccade": 213,
        "R_saccade": 214,
        "L_blink": 215,
        "R_blink": 216,
    }

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

    def to_mne(self, verbose: bool = False) -> (mne.io.RawArray, Dict[str, int]):
        et_trigs, ses_trigs, stim_trigs = DotsSession._events_df_to_channel(self._events, self.num_samples)

        # create mapping from event name to event code
        event_dict = dict()
        event_dict.update({k: np.uint(v) for k, v in DotsSession.__EVENTS_DICT.items()})
        event_dict.update({f"stim_{val}": val for val in np.unique(stim_trigs) if val != 0})
        if not all(np.isin(np.unique(et_trigs[et_trigs != 0]), list(event_dict.values()))):
            raise AssertionError("Unexpected event code in ET triggers")
        if not all(np.isin(np.unique(ses_trigs[ses_trigs != 0]), list(event_dict.values()))):
            raise AssertionError("Unexpected event code in session triggers")
        if not all(np.isin(np.unique(stim_trigs[stim_trigs != 0]), list(event_dict.values()))):
            raise AssertionError("Unexpected event code in stim triggers")

        # create MNE RawArray object
        chanlocs = self.get_channel_locations()
        info = mne.create_info(
            ch_names=chanlocs['labels'].tolist() + ['TRIGGER_ET', 'TRIGGER_SES', 'TRIGGER_STIM'],
            ch_types=chanlocs['type'].tolist() + ['stim', 'stim', 'stim'],
            sfreq=self.sampling_rate,
        )
        raw = mne.io.RawArray(
            np.vstack((self.get_data(), et_trigs, ses_trigs, stim_trigs)), info, verbose=verbose
        )
        return raw, event_dict

    @property
    def session_num(self) -> int:
        return self._session_num

    @staticmethod
    def _events_from_dict(events: Dict[str, list]) -> pd.DataFrame:
        events_df = pd.DataFrame(events)
        missing_columns = set(DotsSession._EVENT_COLUMNS) - set(events_df.columns)
        if missing_columns:
            raise ValueError(f"Missing columns in events DataFrame: {missing_columns}")

        # calculate end time for each event (including 0-duration events)
        new_endtime = events_df['latency'] + events_df['duration']
        is_zero_dur = events_df['duration'] == 0
        new_endtime[~is_zero_dur] -= 1      # start count from zero so subtract 1 from end time
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

        # parse the "type" column
        events_df['orig_type'] = events_df['type']
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

        # replace empty array cells with NaN
        channel_locs_df = channel_locs_df.map(
            lambda val: np.nan if isinstance(val, np.ndarray) and len(val) == 0 else val
        )
        return channel_locs_df

    @staticmethod
    def _events_df_to_channel(
            events_df: pd.DataFrame, n_samples: int = None
    ) -> (np.ndarray, np.ndarray, np.ndarray):
        if n_samples is None:
            n_samples = int(events_df['endtime'].max())
        if n_samples <= 0:
            raise ValueError("n_samples must be a positive integer")
        is_zero_duration = events_df['duration'] == 0
        is_string_event_type = events_df['type'].map(lambda x: isinstance(x, str))
        non_zero_string_events = events_df.loc[~is_zero_duration & is_string_event_type]
        is_valid = non_zero_string_events['type'].map(
            lambda typ: "fixation" in typ.lower() or "saccade" in typ.lower() or "blink" in typ.lower()
        ).all()
        if not is_valid:
            raise AssertionError("Unexpected event type in events DataFrame")

        # replace the 'type' column with integer codes
        new_events = events_df.copy().sort_values('latency')
        new_events['type'] = new_events['type'].map(
            lambda typ: typ if type(typ) in [int, float, np.uint] else DotsSession.__EVENTS_DICT.get(typ, np.nan)
        )
        is_nan = new_events['type'].isna().any()
        if is_nan:
            raise AssertionError("Unexpected event type in events DataFrame")

        # populate the trigger channel
        et_trigs = np.zeros(n_samples, dtype=np.uint8)  # max 255 events (excluding 0)
        ses_trigs = np.zeros(n_samples, dtype=np.uint8)  # max 255 events (excluding 0)
        stim_trigs = np.zeros(n_samples, dtype=np.uint8)  # max 255 events (excluding 0)
        et_evnt_codes = [
            # ET events are encoded as 211-216
            v for k, v in DotsSession.__EVENTS_DICT.items()
            if "fixation" in k.lower() or "saccade" in k.lower() or "blink" in k.lower()
        ]
        session_evnt_codes = [
            # Session events (stim_off, block_on, block_off, etc.) are encoded as 41, 55, 56, 201-205
            v for k, v in DotsSession.__EVENTS_DICT.items()
            if v not in et_evnt_codes
        ]
        for evnt in new_events['type'].unique():
            trigs_idxs = np.arange(n_samples)
            is_evnt = new_events['type'] == evnt
            is_event_idx = np.any(
                    (trigs_idxs >= new_events.loc[is_evnt, 'latency'].to_numpy()[:, None]) &
                    (trigs_idxs <= new_events.loc[is_evnt, 'endtime'].to_numpy()[:, None]),
                    axis=0
            )
            # populate the correct trigger channel
            if evnt in et_evnt_codes:
                et_trigs[is_event_idx] = np.uint8(evnt)
            elif evnt in session_evnt_codes:
                ses_trigs[is_event_idx] = np.uint8(evnt)
            else:
                stim_trigs[is_event_idx] = np.uint8(evnt)
        return et_trigs, ses_trigs, stim_trigs

    @staticmethod
    def __parse_event_types(event_type: pd.Series) -> pd.Series:
        event_type = event_type.map(
            lambda val: str(val).strip()
        ).map(
            lambda val: int(val) if val.isnumeric() else val
        )
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
        block_type = pd.Series(DotsBlockTaskType.OUT_OF_BLOCK, index=event_type.index)
        for i, (on, off) in enumerate(zip(block_on_idxs, block_off_idxs)):
            block_type.loc[on:off] = DotsBlockTaskType(i + 1)
        return block_type

    def __repr__(self):
        return f"{super().__repr__()}{self.session_num}"
