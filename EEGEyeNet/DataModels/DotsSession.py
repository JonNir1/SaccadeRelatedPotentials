import os
from typing import Dict, Tuple, Union
from enum import IntEnum

import numpy as np
import pandas as pd
import mne

from EEGEyeNet.DataModels.BaseSession import BaseSession, SessionTaskType


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
    __ORIGINAL_TARGET_LOCATIONS: Dict[int, Tuple[int, int]] = {
        # dot locations specified in EEGEyeNet's OSF appendix on experimental paradigms (https://osf.io/ktv7m/)
        # IMPORTANT NOTE: the (x,y) coordinates follow Matlab convention, so bottom-left is (0,0) and top-right is (800,600)
        1: (400, 300),  # middle
        2: (650, 500), 3: (400, 100), 4: (100, 450), 5: (700, 450), 6: (100, 500), 7: (200, 350), 8: (300, 400),
        9: (100, 150), 10: (150, 500), 11: (150, 100), 12: (700, 100), 13: (300, 200), 14: (100, 100), 15: (700, 500),
        16: (500, 400), 17: (600, 250), 18: (650, 100),
        19: (400, 300),  # middle
        20: (200, 250), 21: (400, 500), 22: (700, 150), 23: (500, 200), 24: (100, 300), 25: (700, 300), 26: (600, 350),
        27: (400, 300),  # middle
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
        data, timestamps, events, channel_locs, ref = DotsSession._parse_mat_file(path)
        # post-process the `events` DataFrame
        events['orig_type'] = events['type']
        events['type'] = DotsSession.__parse_event_types(events['type'])
        events['block'] = DotsSession.__extract_block_type_event(
            events['type'])  # add "block" column: basic -> reversed -> mirrored -> reversed_mirrored -> basic2

        # extract metadata from path
        basename = os.path.basename(path)  # example: EP12_DOTS5_EEG.mat
        subject_id = basename.split("_")[0].capitalize()
        session_num = int(basename.split("_")[1][-1])

        # check if grid number in events matches metadata
        _grid_num = int((events['type'][
            events['type'].map(lambda val: str(val).startswith(DotsSession.__GRID_PREFIX_STR))
        ].iloc[0])[-1])
        if _grid_num != session_num:
            raise AssertionError(f"Grid number in events ({_grid_num}) must match metadata ({session_num})")

        # return DotsSession object
        ses = DotsSession(subject_id, data, timestamps, events, channel_locs, session_num, ref)
        return ses

    def to_mne(self, reog_ref: Union[str, int] = 'Pz', verbose: bool = False) -> (mne.io.RawArray, Dict[str, int]):
        et_triggers, ses_triggers, dot_triggers = DotsSession._events_df_to_channel(self._events, self.num_samples)

        # create mapping from event name to event code
        event_dict = dict()
        event_dict.update({k: np.uint(v) for k, v in DotsSession.__EVENTS_DICT.items()})
        event_dict.update({f"stim_{val}": val for val in np.unique(dot_triggers) if val != 0})
        if not all(np.isin(np.unique(et_triggers[et_triggers != 0]), list(event_dict.values()))):
            raise AssertionError("Unexpected event code in ET triggers")
        if not all(np.isin(np.unique(ses_triggers[ses_triggers != 0]), list(event_dict.values()))):
            raise AssertionError("Unexpected event code in session triggers")
        if not all(np.isin(np.unique(dot_triggers[dot_triggers != 0]), list(event_dict.values()))):
            raise AssertionError("Unexpected event code in stim triggers")

        # create MNE RawArray object
        chanlocs = self.get_channel_locations()
        info = mne.create_info(
            ch_names=chanlocs['labels'].tolist() + ['rEOG'] + ['STI_ET', 'STI_SES', 'STI_DOT'],
            ch_types=chanlocs['type'].tolist() + ['eog'] + ['stim', 'stim', 'stim'],
            sfreq=self.sampling_rate,
        )
        raw = mne.io.RawArray(
            np.vstack((self.get_data(), self.calculate_radial_eog(reog_ref), et_triggers, ses_triggers, dot_triggers)),
            info,
            verbose=verbose
        )
        return raw, event_dict

    @property
    def session_num(self) -> int:
        return self._session_num

    @staticmethod
    def get_dot_locations(block: Union[DotsBlockTaskType, int, str]) -> pd.DataFrame:
        """
        Get the target locations in Python coordinates (origin at top-left) for a given block number.
        Blocks follow the order: 1=basic -> 2=reversed -> 3=mirrored -> 4=reversed_mirrored -> 5=basic2. Other block
        types raise a ValueError.
        TECHNICAL NOTE: EEGEyeNet uses a 800x600 screen resolution

        :param block: the block type (DotsBlockTaskType), its code (int), or its name (str)
        :returns: a DataFrame is indexed by dot appearance and contains the following columns:
            - 'x' and 'y' representing the horizontal and vertical coordinates (Python standard), respectively.
            - 'angle2center' representing the angle in degrees from the center of the screen (N=0, W=90, S=180, E=-90).
            - 'pixel_distance' representing the distance from the previous dot in pixels.
            - 'azimuth' representing the angle in degrees from the previous dot (N=0, W=90, S=180, E=-90).
        """
        if isinstance(block, str):
            block = DotsBlockTaskType[block.upper().strip()]
        elif isinstance(block, int):
            block = DotsBlockTaskType(block)
        if not isinstance(block, DotsBlockTaskType):
            raise TypeError("block must be a DotsBlockTaskType, its code (int 1-5), or its name (str)")
        if block == DotsBlockTaskType.OUT_OF_BLOCK:
            raise ValueError("Cannot get dot locations for OUT_OF_BLOCK block type")

        # get dot locations based on the block type
        base_locations = pd.DataFrame(DotsSession.__ORIGINAL_TARGET_LOCATIONS).T
        base_locations.columns = ['x', 'y']
        base_locations['y'] = 600 - base_locations['y']  # convert to Python coordinates, origin at top-left
        if block == DotsBlockTaskType.BASIC or block == DotsBlockTaskType.BASIC2:
            locations = base_locations
        elif block == DotsBlockTaskType.REVERSED:
            locations = base_locations.iloc[::-1].reset_index(drop=True)
        elif block == DotsBlockTaskType.MIRRORED:
            locations = base_locations.copy()
            locations['x'] = 800 - locations['x']
            locations['y'] = 600 - locations['y']
        elif block == DotsBlockTaskType.REVERSED_MIRRORED:
            locations = base_locations.iloc[::-1].reset_index(drop=True)
            locations['x'] = 800 - locations['x']
            locations['y'] = 600 - locations['y']
        else:
            raise ValueError(f"Unexpected block type: {block}")

        # calculate additional values for each dot location
        center_x, center_y = 400, 300  # center of the screen in EEGEyeNet's apparatus
        locations['angle2center'] = np.arctan2(
            center_y - locations['y'], locations['x'] - center_x
        ) * 180 / np.pi - 90  # subtract 90 to make N=0, W=90, S=180, E=-90
        dx, dy = locations['x'].diff(), -1 * locations['y'].diff()  # negative y because y increases downwards
        locations['pixel_distance'] = np.sqrt(dx ** 2 + dy ** 2)
        locations['azimuth'] = np.arctan2(dy, dx) * 180 / np.pi - 90  # subtract 90 to make N=0, W=90, S=180, E=-90
        return locations

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
        new_events['type'] = new_events['type'].replace(
            # replace events "19" and "27" with "1"; and "119" and "127" with "101" - all represent the middle dot
            {1: 1, 19: 1, 27: 1, 101: 101, 119: 101, 127: 101}
        )

        # populate the trigger channels
        et_triggers = np.zeros(n_samples, dtype=np.uint8)  # max 255 events (excluding 0)
        ses_triggers = np.zeros(n_samples, dtype=np.uint8)  # max 255 events (excluding 0)
        dot_triggers = np.zeros(n_samples, dtype=np.uint8)  # max 255 events (excluding 0)
        et_evnt_codes = [
            # ET events are encoded as 211-216
            v for k, v in DotsSession.__EVENTS_DICT.items()
            if "fixation" in k.lower() or "saccade" in k.lower() or "blink" in k.lower()
        ]
        session_evnt_codes = [
            # Session events (block_on, block_off, etc.) are encoded as 55, 56, 201-205
            v for k, v in DotsSession.__EVENTS_DICT.items()
            if (v not in et_evnt_codes) and (k != "stim_off")   # exclude stim_off (code 41)
        ]
        idxs = np.arange(n_samples)
        for evnt in new_events['type'].unique():
            is_evnt = new_events['type'] == evnt
            is_event_idx = np.any(
                    (idxs >= new_events.loc[is_evnt, 'latency'].to_numpy()[:, None]) &
                    (idxs <= new_events.loc[is_evnt, 'endtime'].to_numpy()[:, None]),
                    axis=0
            )
            # populate the correct trigger channel
            if evnt in et_evnt_codes:
                et_triggers[is_event_idx] = np.uint8(evnt)
            elif evnt in session_evnt_codes:
                ses_triggers[is_event_idx] = np.uint8(evnt)
            else:
                dot_triggers[is_event_idx] = np.uint8(evnt)
        return et_triggers, ses_triggers, dot_triggers

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
    def __extract_block_type_event(event_type: pd.Series) -> pd.Series:
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
        return f"{super().__repr__()}_{self.session_num}"
