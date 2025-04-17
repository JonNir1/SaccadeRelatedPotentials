import os
import warnings
from typing import Dict, Tuple, List
from enum import IntEnum
from numbers import Number
import pickle as pkl

import numpy as np
import pandas as pd
import mne

from EEGEyeNet.DataModels.BaseRecording import BaseRecording, TaskType

_BLOCK_PREFIX_STR: str = "block_"
_EVENTS_DICT = {
    "stim_off": 41,
    "grid_on": 55,
    "grid_off": 56,

    f"{_BLOCK_PREFIX_STR}1": 201,
    f"{_BLOCK_PREFIX_STR}2": 202,
    f"{_BLOCK_PREFIX_STR}3": 203,
    f"{_BLOCK_PREFIX_STR}4": 204,
    f"{_BLOCK_PREFIX_STR}5": 205,
    f"{_BLOCK_PREFIX_STR}6": 206,

    "L_fixation": 211,
    "R_fixation": 212,
    "L_saccade": 213,
    "R_saccade": 214,
    "L_blink": 215,
    "R_blink": 216,
}
_DOT_LOCATIONS: Dict[int, Tuple[int, int]] = {
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


class DotsGridType(IntEnum):
    OUT_OF_GRID = 0
    BASIC = 1
    REVERSED = 2
    MIRRORED = 3
    REVERSED_MIRRORED = 4
    BASIC2 = 5


class DotsBlock(BaseRecording):
    """
    Class representing a block of the EEGEyeNet `Dots` task. A block is a single recording of a single subject's data,
    where the subject performed EEGEyeNet's `Dots` (aka `Large Grid`) task. Each block consists of five "grids" of dots,
    where each grid is a collection of dots that are presented to the subject in a specific order, based on the
    `DotsGridType` enum.
    """
    _TASK_TYPE = TaskType.DOTS

    def __init__(
            self,
            subject: str,
            data: np.ndarray,
            timestamps: np.ndarray,
            events: pd.DataFrame,
            channel_locations: pd.DataFrame,
            block_num: int,
            reference: str = "average"
    ):
        super().__init__(subject, data, timestamps, events, channel_locations, reference)
        self._block_num = block_num

    @staticmethod
    def from_mat_file(path: str) -> "DotsBlock":
        data, timestamps, events, channel_locs, ref = DotsBlock._parse_mat_file(path)

        # extract metadata from path
        basename = os.path.basename(path)  # example: EP12_DOTS5_EEG.mat
        subject_id = basename.split("_")[0].capitalize()
        block_num = int(basename.split("_")[1][-1])

        # post-process the `events` DataFrame
        events = DotsBlock._post_process_events(events, block_num)

        # return DotsBlock object
        ses = DotsBlock(subject_id, data, timestamps, events, channel_locs, block_num, ref)
        return ses

    @property
    def block_num(self) -> int:
        return self._block_num

    def to_mne(self, verbose: bool = False) -> (mne.io.RawArray, Dict[str, int]):
        # create mapping from event-name to event-code
        et_triggers, ses_triggers, dot_triggers = DotsBlock.__events_df_to_mne_channels(
            self._events, self.num_samples
        )
        event_dict = dict()
        for k, v in _EVENTS_DICT.items():
            k1, k2 = k.lower().split("_")
            if k2 in ["fixation", "saccade", "blink"]:
                event_dict[f"{k2}/{k1}"] = np.uint8(v)  # key changes from "L_Fixation" to "fixation/l"
            else:
                event_dict[f"{k1}/{k2}"] = np.uint8(v)  # key changes from "grid_on" to "grid/on"
        event_dict.update({
            f"stim/{val}": val for val in np.unique(dot_triggers)
            if val not in {0, _EVENTS_DICT['stim_off']}
        })
        if not all(np.isin(np.unique(et_triggers[et_triggers != 0]), list(event_dict.values()))):
            raise AssertionError("Unexpected event code in ET triggers")
        if not all(np.isin(np.unique(ses_triggers[ses_triggers != 0]), list(event_dict.values()))):
            raise AssertionError("Unexpected event code in grid triggers")
        if not all(np.isin(np.unique(dot_triggers[dot_triggers != 0]), list(event_dict.values()))):
            raise AssertionError("Unexpected event code in stim triggers")

        # create MNE Info object
        chan_locs = self.get_channel_locations()
        info = mne.create_info(
            ch_names=chan_locs['labels'].tolist() + ['STI_ET', 'STI_SES', 'STI_DOT'],
            ch_types=chan_locs['type'].tolist() + ['stim', 'stim', 'stim'],
            sfreq=self.sampling_rate,
        )

        # create MNE RawArray object
        unit_conversion = pd.Series(info.get_channel_types()).map(
            dict(
                eeg=1e-6, eog=1e-6,  # EEGEyeNet records uV but MNE requires eeg/eog channels in V
                eyegaze=1, pupil=1,  # eye gaze in pixels; pupil in AU
                misc=1,  # ET timestamps in ms
                stim=1,  # triggers are in raw values
            )
        )
        channels = np.multiply(
            np.vstack((self.get_data(as_frame=False), et_triggers, ses_triggers, dot_triggers)),
            unit_conversion.values[:, np.newaxis]
        )
        raw = mne.io.RawArray(channels, info, verbose=verbose).drop_channels('ET_TIME', on_missing='ignore')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            raw.set_montage("GSN-HydroCel-128", on_missing='ignore', verbose=verbose)

        # set channel types for eyetracking data
        eyetracking_channel_mapping = {
            'L-GAZE-X': ('eyegaze', 'px', 'left', 'x'), 'L-GAZE-Y': ('eyegaze', 'px', 'left', 'y'),
            'R-GAZE-X': ('eyegaze', 'px', 'right', 'x'), 'R-GAZE-Y': ('eyegaze', 'px', 'right', 'y'),
            'L-AREA': ('pupil', 'au', 'left'), 'R-AREA': ('pupil', 'au', 'right'),
        }
        is_eyetracking_channel = np.isin(raw.get_channel_types(), ['eyegaze', 'pupil'])
        eyetracking_channel_names = np.array(raw.ch_names)[is_eyetracking_channel]
        eyetracking_channel_mapping = {
            k: v for k, v in eyetracking_channel_mapping.items() if k in eyetracking_channel_names
        }
        raw = mne.preprocessing.eyetracking.set_channel_types_eyetrack(raw, mapping=eyetracking_channel_mapping)
        return raw, event_dict

    @staticmethod
    def get_dot_coordinates(dot_number: int) -> Tuple[int, int]:
        """
        Get the (x, y) coordinates of the specified dot number.
        NOTE: the origin in the bottom-left corner of the screen.
        """
        if 1 <= dot_number <= 27:
            base_x, base_y = _DOT_LOCATIONS.get(dot_number, (None, None))
        elif 101 <= dot_number <= 127:
            # sometimes dots are marked with `101` instead of `1`
            base_x, base_y = _DOT_LOCATIONS.get(dot_number - 100, (None, None))
        else:
            raise KeyError(f"Invalid dot number: {dot_number}")
        if base_x is None or base_y is None:
            raise ValueError(f"Unknown coordinates for dot number {dot_number}.")
        return base_x, base_y

    @staticmethod
    def _post_process_events(events: pd.DataFrame, block_num: int) -> pd.DataFrame:
        # parse event types
        events['orig_type'] = events['type']
        events['type'] = DotsBlock.__parse_event_types(events['type'])

        # extract grid type: basic -> reversed -> mirrored -> reversed_mirrored -> basic2
        events['grid'] = DotsBlock.__extract_grid_type_event(events['type'])

        # append dots metadata (coordinates, etc.)
        events = DotsBlock.__append_dots_metadata(events)

        # assert that block number in events matches block number from metadata
        _block_num = int((events['type'][events['type'].map(
            lambda val: str(val).startswith(_BLOCK_PREFIX_STR)
        )].iloc[0])[-1])
        if _block_num != block_num:
            raise AssertionError(f"Grid number in events ({_block_num}) must match metadata ({block_num})")
        return events

    @staticmethod
    def __parse_event_types(event_type: pd.Series) -> pd.Series:
        event_type = event_type.map(lambda val: str(val).strip())
        event_type = event_type.map(lambda val: int(val) if val.isnumeric() else val)
        event_type = event_type.map(
            lambda val: val - 100 if isinstance(val, int) and 100 <= val < 200 else val  # convert 101-127 to 1-27
        )
        event_type = event_type.replace({41: "stim_off", 55: "grid_on", 56: "grid_off"})

        # find grid number: the first event labelled 12-17 before the first grid_on event
        grid_on_idxs = event_type[event_type == "grid_on"].index
        events_preceding_grid = event_type.iloc[:grid_on_idxs.min()]
        is_block_num_event = np.isin(events_preceding_grid, np.arange(12, 18))
        if is_block_num_event.sum() == 0:
            raise ValueError("No block number event found before first `grid_on` event")
        if is_block_num_event.sum() > 1:
            raise ValueError("Multiple block number events found before first `grid_on` event")
        block_num_idx = events_preceding_grid[is_block_num_event].index.min()
        block_number = int(event_type[block_num_idx]) - 11  # blocks 1-6 are labelled 12-17
        event_type[block_num_idx] = f"{_BLOCK_PREFIX_STR}{block_number}"
        return event_type

    @staticmethod
    def __extract_grid_type_event(event_type: pd.Series) -> pd.Series:
        grid_on_idxs = event_type[(event_type == "grid_on") | (event_type == 55)].index
        grid_off_idxs = event_type[(event_type == "grid_off") | (event_type == 56)].index
        if len(grid_on_idxs) != len(grid_off_idxs):
            raise ValueError("Number of `grid_on` and `grid_off` events must match")
        if len(grid_on_idxs) != 5:
            raise ValueError(f"Number of grids must be 5, found {len(grid_on_idxs)}")
        grid_type = pd.Series(DotsGridType.OUT_OF_GRID, index=event_type.index)
        for i, (on, off) in enumerate(zip(grid_on_idxs, grid_off_idxs)):
            grid_type.loc[on:off] = DotsGridType(i + 1)
        return grid_type

    @staticmethod
    def __append_dots_metadata(events: pd.DataFrame) -> pd.DataFrame:
        """
        For each of the displayed stimuli (dots), append the following columns:
        - 'stim_x' and 'stim_y' representing the horizontal and vertical coordinates, respectively (left-bottom origin).
        - 'center_distance_px' representing the distance in pixels of the current dot from the center of the screen.
        - 'center_angle_deg' representing the angle in degrees from the center of the screen (N=0, W=90, S=180, E=-90).
        - 'prev_distance_px' representing the distance in pixels of the current dot from the previous dot in the block.
        - 'prev_angle_deg' representing the angle in degrees from the previous dot (N=0, W=90, S=180, E=-90).

        NOTE: EEGEyeNet follows the Matlab convention for screen coordinates, where bottom-left is (0,0).

        :param events: the DataFrame containing the events data
        :returns: the DataFrame with the appended columns
        """
        is_dot_event = ~np.isin(events['type'], list(_EVENTS_DICT.keys()))

        # dot locations
        events.loc[is_dot_event, 'stim_x'] = events.loc[is_dot_event, 'type'].map(
            lambda dot_id: DotsBlock.get_dot_coordinates(dot_id)[0]
        )
        events.loc[is_dot_event, 'stim_y'] = events.loc[is_dot_event, 'type'].map(
            lambda dot_id: DotsBlock.get_dot_coordinates(dot_id)[1]
        )

        # relative to screen center
        screen_resolution = (800, 600)  # EEGEyeNet uses 800x600 screen resolution
        center_x, center_y = screen_resolution[0] / 2, screen_resolution[1] / 2
        events.loc[is_dot_event, 'center_distance_px'] = np.sqrt(
            (events.loc[is_dot_event, 'stim_x'] - center_x) ** 2 + (events.loc[is_dot_event, 'stim_y'] - center_y) ** 2
        )
        center_angle = np.arctan2(
            events.loc[is_dot_event, 'stim_y'] - center_y,
            events.loc[is_dot_event, 'stim_x'] - center_x
        ) * 180 / np.pi - 90  # subtract 90 to make N=0, W=90, S=180, E=-90
        # conform angles to range (-180, 180] where North is 0 degrees
        center_angle[center_angle <= -180] += 360
        center_angle[center_angle > 180] -= 360
        events.loc[is_dot_event, 'center_angle_deg'] = center_angle
        events.loc[events['center_distance_px'] == 0, 'center_angle_deg'] = 0  # correct angle for zero distance

        # relative to previous dot (1st dot in each block should get `NaN`)
        for g in events['grid'].unique():
            is_block = events['grid'] == g
            is_dot_in_block = is_dot_event & is_block
            dx = events.loc[is_dot_in_block, 'stim_x'].diff()
            dy = events.loc[is_dot_in_block, 'stim_y'].diff()
            events.loc[is_dot_in_block, 'prev_distance_px'] = np.sqrt(dx ** 2 + dy ** 2)
            prev_angle = np.arctan2(dy, dx) * 180 / np.pi - 90  # subtract 90 to make N=0, W=90, S=180, E=-90
            # conform angles to range (-180, 180] where North is 0 degrees
            prev_angle[prev_angle <= -180] += 360
            prev_angle[prev_angle > 180] -= 360
            events.loc[is_dot_in_block, 'prev_angle_deg'] = prev_angle
            events.loc[events['prev_distance_px'] == 0, 'center_angle_deg'] = 0  # correct angle for zero distance
        return events

    @staticmethod
    def __events_df_to_mne_channels(
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
            lambda typ: typ if isinstance(typ, Number) else _EVENTS_DICT.get(typ, np.nan)
        )
        is_nan = new_events['type'].isna().any()
        if is_nan:
            raise AssertionError("Unexpected event type in events DataFrame")
        new_events['type'] = new_events['type'].replace(
            # TODO: add the previous dot to the event code for future analysis (e.g. "2\14")
            # replace events "19" and "27" with "1"; and "119" and "127" with "101" - all represent the middle dot
            {1: 1, 19: 1, 27: 1, 101: 101, 119: 101, 127: 101}
        )

        # populate the trigger channels
        et_triggers = np.zeros(n_samples, dtype=np.uint8)  # max 255 events (excluding 0)
        ses_triggers = np.zeros(n_samples, dtype=np.uint8)  # max 255 events (excluding 0)
        dot_triggers = np.zeros(n_samples, dtype=np.uint8)  # max 255 events (excluding 0)
        et_evnt_codes = [
            # ET events are encoded as 211-216
            v for k, v in _EVENTS_DICT.items()
            if "fixation" in k.lower() or "saccade" in k.lower() or "blink" in k.lower()
        ]
        block_evnt_codes = [
            # Block events (block_on, block_off, etc.) are encoded as 55, 56, 201-205
            v for k, v in _EVENTS_DICT.items()
            if (v not in et_evnt_codes) and (k != "stim_off") and (v != 41)  # exclude stim_off (code 41)
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
            elif evnt in block_evnt_codes:
                ses_triggers[is_event_idx] = np.uint8(evnt)
            else:
                dot_triggers[is_event_idx] = np.uint8(evnt)
        return et_triggers, ses_triggers, dot_triggers

    def __repr__(self):
        return f"{super().__repr__()}_{self.block_num}"


class DotsSession:
    """
    Class representing a session of the EEGEyeNet `Dots` task. A session is one subject's data, cosisting of multiple
    blocks represented as `DotsBlock` objects.
    """
    __EXPECTED_BLOCK_COUNT = len([k for k, v in _EVENTS_DICT.items() if k.startswith(_BLOCK_PREFIX_STR)])

    def __init__(self, blocks: List[DotsBlock]):
        types = {type(block) for block in blocks}
        if len(types) != 1 or DotsBlock not in types:
            raise TypeError("All blocks must be of type DotsBlock")
        subjects = {block.subject for block in blocks}
        if len(subjects) != 1:
            raise ValueError("All DotBlock objects must belong to the same subject")
        blocks = sorted(blocks, key=lambda b: b.block_num, reverse=False)
        self._blocks = blocks
        self._subject = blocks[0].subject

    @property
    def subject(self) -> str:
        return self._subject

    @property
    def num_blocks(self) -> int:
        return len(self._blocks)

    @property
    def missing_blocks(self) -> List[int]:
        """ Get the list of missing blocks in the session, or empty list if all blocks are present. """
        all_block_nums = set(range(1, self.__EXPECTED_BLOCK_COUNT + 1))
        present_block_nums = {block.block_num for block in self._blocks}
        return sorted(list(all_block_nums - present_block_nums))

    def get_block(self, block_num: int) -> DotsBlock:
        """
        Get the block with the specified block number.
        :param block_num: the block number
        :return: the DotsBlock object
        """
        for block in self._blocks:
            if block.block_num == block_num:
                return block
        raise KeyError(f"Block {block_num} not found in Session.")

    def to_mne(self) -> (mne.io.RawArray, Dict[str, int]):
        # TODO: implement this method
        raise NotImplementedError("DotsSession.to_mne() is not implemented. Use DotsBlock.to_mne() instead.")

    def to_pickle(self, path: str):
        """
        Save the DotsSession object to a pickle file.
        :param path: the path to save the pickle file
        """
        with open(path, 'wb') as f:
            pkl.dump(self, f)

    @staticmethod
    def from_pickle(path: str) -> "DotsSession":
        """
        Load the DotsSession object from a pickle file.
        :param path: the path to the pickle file
        :return: the DotsSession object
        """
        with open(path, 'rb') as f:
            return pkl.load(f)

    def __eq__(self, other):
        if not isinstance(other, DotsSession):
            return False
        if self.subject != other.subject:
            return False
        if self.num_blocks != other.num_blocks:
            return False
        if self.missing_blocks != other.missing_blocks:
            return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return f"{self.__class__.__name__}_{self._subject.upper()}"

    def __str__(self):
        return self.__repr__()

