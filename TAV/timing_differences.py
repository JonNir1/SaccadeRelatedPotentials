import os
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm
import plotly.graph_objects as go
import plotly.io as pio

import TAV.tav_helpers as tavh
from TAV.Subject import Subject

_OUTPUT_DIR = tavh.get_output_subdir(os.path.basename(__file__))
_FIGURES_DIR = os.path.join(_OUTPUT_DIR, tavh.constants.FIGURES_STR)

_MAX_DIFF = 20  # maximum difference between GT and Pred event indices to consider them as matched


def load_or_calc_saccade_timing_differences():
    os.makedirs(_FIGURES_DIR, exist_ok=True)
    fname = "saccade_timing_diffs.pkl"
    try:
        diffs = pd.read_pickle(os.path.join(_OUTPUT_DIR, fname))
    except FileNotFoundError:
        onset_diffs, offset_diffs = {}, {}
        for i in tqdm(range(101, 111), desc="Subjects"):
            s = Subject.load_or_make(i, tavh.RESULTS_DIR)
            onset_diffs[i] = saccade_event_detection_sample_difference(s, "saccade_onset")
            # offset_diffs[i] = saccade_event_detection_sample_difference(s, "saccade_offset")
            offset_diffs[i] = np.array([])  # TODO: remove this when offset detection is implemented
        diffs = pd.DataFrame({"saccade_onset": onset_diffs, "saccade_offset": offset_diffs})
        diffs.to_pickle(os.path.join(_OUTPUT_DIR, fname))

    subject_figures = {}
    for idx in tqdm(diffs.index, desc="Subject Figures"):
        fname = f"s{idx}_saccade_timing_diffs"
        try:
            with open(os.path.join(_FIGURES_DIR, fname + ".json"), 'rb') as f:
                fig = pio.read_json(f)
        except FileNotFoundError:
            fig = create_figure(diffs.loc[idx])
            tavh.save_figure(fig, _FIGURES_DIR, fname)
        subject_figures[idx] = fig

    fname = "aggregated_saccade_timing_diffs"
    try:
        with open(os.path.join(_FIGURES_DIR, fname + ".json"), 'rb') as f:
            agg_fig = pio.read_json(f)
    except FileNotFoundError:
        aggregated_diffs = diffs.T.agg(list, axis=1).apply(np.concatenate)
        agg_fig = create_figure(aggregated_diffs)
        tavh.save_figure(agg_fig, _FIGURES_DIR, fname)
    return diffs, subject_figures, agg_fig



def saccade_event_detection_sample_difference(
        s: Subject, event_name: str, max_diff: int = _MAX_DIFF, enforce_trials: bool = True
) -> np.ndarray:
    """
    Calculated the sample-difference between pairs of matching Ground-Truth and Predicted events. Events are matched
    such that the difference between their indices is minimal. The difference is calculated as `t(GT) - t(Pred)`.
    Returns an array of sample differences between matched events.
    """
    # match ET detected events with REOG detected events
    et_event_idxs = s.get_eye_tracking_event_indices(event_name, False)
    et_event_channel = s.create_boolean_event_channel(et_event_idxs, enforce_trials)
    reog_event_idxs = s.calculate_reog_saccade_event_indices(
        event_name=event_name, filter_name='srp', snr=3.5, enforce_trials=False
    )
    reog_event_channel = s.create_boolean_event_channel(reog_event_idxs, enforce_trials)
    all_matched_idxs = tavh.match_boolean_events(et_event_channel, reog_event_channel)
    # calculate time difference between matched events
    all_match_diffs = np.diff(all_matched_idxs, axis=1).flatten()
    allowed_diffs = all_match_diffs[abs(all_match_diffs) <= max_diff]
    return allowed_diffs


def create_figure(sample_diffs: pd.Series, colormap: List[str] = None,) -> go.Figure:
    colormap = colormap or tavh.constants.DEFAULT_COLORMAP
    fig = go.Figure()
    for i, evnt in enumerate(sample_diffs.index):
        event_diffs = sample_diffs[evnt]
        if event_diffs is None or len(event_diffs) == 0:
            continue
        event_type = evnt.split("_")[0].lower()
        fig.add_trace(
            go.Violin(
                x=[event_type] * len(event_diffs),
                y=event_diffs,
                name=evnt,
                legendgroup=evnt,
                scalegroup=evnt,
                side='positive' if i % 2 == 0 else 'negative',
                line_color=colormap[i],
                meanline=dict(visible=True),
                spanmode='hard',
            )
        )
    fig.update_layout(
        title=f"Event Detection Timing Differences - Subject {sample_diffs.name}",
        yaxis_title="Sample Difference",
        showlegend=True,
    )
    return fig


