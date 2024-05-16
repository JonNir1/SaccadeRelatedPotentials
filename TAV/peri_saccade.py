import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import pickle as pkl
from tqdm import tqdm
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio

import TAV.tav_helpers as tavh
from TAV.Subject import Subject


_OUTPUT_DIR = tavh.get_output_subdir(os.path.basename(__file__))
_FIGURES_DIR = os.path.join(_OUTPUT_DIR, tavh.FIGURES_STR)
_LINE_FIGURES_DIR = os.path.join(_FIGURES_DIR, "lines")
_ACTIVITY_FIGURES_DIR = os.path.join(_OUTPUT_DIR, tavh.FIGURES_STR, "offset_activity")
_DEFAULT_COLORMAP = px.colors.qualitative.Dark24

_SAMPLES_BEFORE, _SAMPLES_AFTER = 250, 250
ELECTRODES = {"F4": "#5aae61", "FC4": "#1b7837", "O2": "#00441b"}
IGNORE_ELECTRODES = {
    'LHEOG', 'RHEOG', 'RVEOGS', 'RVEOGI', 'M1', 'M2', 'LVEOGI', 'ANA1', 'ANA2', 'ANA3', 'ANA4', 'ANA5', 'ANA6', 'ANA7',
    'ANA8', 'PHOTODIODE', 'RESPONSEBOX'
}

###########################


def calculate_or_load_peri_saccade_reog_activity():
    os.makedirs(_LINE_FIGURES_DIR, exist_ok=True)
    os.makedirs(_ACTIVITY_FIGURES_DIR, exist_ok=True)
    try:
        with open(os.path.join(_OUTPUT_DIR, "peri_saccade_reog_activity.pkl"), "rb") as f:
            subject_activities = pkl.load(f)
    except FileNotFoundError:
        subject_activities = {}
        for i in tqdm(range(101, 111), desc="Subjects"):
            s = Subject.load_or_make(i, tavh.OUTPUT_DIR)
            activity = calculate_peri_saccade_reog_activity(s)
            subject_activities[i] = activity
        with open(os.path.join(_OUTPUT_DIR, "peri_saccade_reog_activity.pkl"), "wb") as f:
            pkl.dump(subject_activities, f)

    try:
        with open(os.path.join(_LINE_FIGURES_DIR, "reog_figures.pkl"), "rb") as f:
            subject_figures = pkl.load(f)
    except FileNotFoundError:
        subject_figures = {}
        for idx, activity in tqdm(subject_activities.items(), desc="Subject Figures"):
            figs = {}
            for evt in activity.columns:
                line_fig = _peri_event_line_figure(activity[evt], f"Subject {idx}", show_error=False)
                tavh.save_figure(line_fig, _LINE_FIGURES_DIR, f"{evt}_subject_{idx}")
                figs[evt] = line_fig
            subject_figures[idx] = figs
        with open(os.path.join(_LINE_FIGURES_DIR, "reog_figures.pkl"), "wb") as f:
            pkl.dump(subject_figures, f)

    subj_activity = subject_activities[101]  # example subject
    try:
        mean_figs = {}
        for evnt in subj_activity.columns:
            with open(os.path.join(_LINE_FIGURES_DIR, f"{evnt}_mean_subject.json"), 'rb') as f:
                mean_figs[evnt] = pio.read_json(f)
    except FileNotFoundError:
        mean_figs = {}
        for evnt in subj_activity.columns:
            event_data = {}
            for ch_name in subj_activity.index:
                event_data[ch_name] = pd.concat([
                    subj_act[evnt][ch_name] for subj_act in subject_activities.values()
                ])
            event_data = pd.Series(event_data)
            event_data.name = evnt
            event_fig = _peri_event_line_figure(event_data, "Mean Subject", show_error=False)
            tavh.save_figure(event_fig, _LINE_FIGURES_DIR, f"{evnt}_mean_subject")
            mean_figs[evnt] = event_fig

    try:
        with open(os.path.join(_ACTIVITY_FIGURES_DIR, "reog_figures.pkl"), "rb") as f:
            offset_activity_figs = pkl.load(f)
    except FileNotFoundError:
        offset_activity_figs = {}
        channel, evt = "reog", "saccade_offset"
        for idx in tqdm(subject_activities.keys(), desc="Offset Activity Figures"):
            offset_activity = subject_activities[idx].loc[channel, evt]
            activity_fig = _peri_event_image(
                offset_activity,
                f"Peri-{evt.replace('_', ' ').title()}<br><sup>Subject: {idx}  Channel: {channel.upper()}</sup>",
                vlines=[-50],
            )
            tavh.save_figure(activity_fig, _ACTIVITY_FIGURES_DIR, f"{channel}_{evt}_subject_{idx}")
            offset_activity_figs[idx] = activity_fig
        with open(os.path.join(_ACTIVITY_FIGURES_DIR, "reog_figures.pkl"), "wb") as f:
            pkl.dump(offset_activity_figs, f)
    return subject_activities, subject_figures, mean_figs, offset_activity_figs


def calculate_peri_saccade_reog_activity(
        s: Subject,
        n_samples_before: int = _SAMPLES_BEFORE,
        n_samples_after: int = _SAMPLES_AFTER,
) -> pd.DataFrame:
    events = {
        "saccade_onset": s.create_boolean_event_channel(
            s.get_eye_tracking_event_indices("saccade_onset"), enforce_trials=True
        ),
        "saccade_offset": s.create_boolean_event_channel(
            s.get_eye_tracking_event_indices("saccade_offset"), enforce_trials=True
        ),
    }
    channels = {
        "reog": s.get_eeg_channel('reog'),
        "reog_filtered": tavh.apply_filter(s.get_eeg_channel('reog'), 'srp'),
    }
    results = {}
    for event, is_event in events.items():
        event_results = {}
        for ch_name, data in channels.items():
            peri_activity = _peri_event_activity(is_event, data, n_samples_before, n_samples_after)
            event_results[ch_name] = peri_activity
        event_results = pd.Series(event_results)
        event_results.name = event
        results[event] = event_results
    results = pd.DataFrame(results)  # events as columns, channels as rows
    return results


def _peri_event_activity(
        is_event: np.ndarray,
        channel: np.ndarray,
        n_samples_before: int,
        n_samples_after: int,
) -> pd.DataFrame:
    assert is_event.size == channel.size, "`is_event` and `channel` must have the same size"
    is_event_idxs = np.where(is_event)[0]
    # pad the signal if the first or last event is too close to the edge
    pad_before = np.maximum(n_samples_before - is_event_idxs[0], 0)
    pad_after = np.maximum(n_samples_after - (channel.size - is_event_idxs[-1]), 0)
    channel = np.pad(channel, (pad_before, pad_after))
    is_event_idxs += pad_before
    # extract peri-event activity
    start_idxs = np.maximum(is_event_idxs - n_samples_before, 0)
    end_idxs = np.minimum(is_event_idxs + n_samples_after, channel.size)
    peri_event_idxs = np.array([np.arange(start, end) for start, end in zip(start_idxs, end_idxs)])
    data = channel[peri_event_idxs]
    result = pd.DataFrame(data, columns=np.arange(-n_samples_before, n_samples_after))
    result.columns.name = "Sample"
    result.index.name = "Event Number"
    return result


###########################


def _peri_event_line_figure(
        data: pd.Series,
        subject_str: str,
        colormap: List[str] = None,
        show_error: bool = False,
):
    event_name = str(data.name).replace("_", " ").title()
    colormap = colormap or _DEFAULT_COLORMAP
    fig = go.Figure()
    for i, channel_name in enumerate(data.index):
        channel_data = data[channel_name]
        timestamps = channel_data.columns
        mean = channel_data.mean(axis=0)
        color = colormap[i % len(colormap)]
        rgb = tuple(int((color.lstrip('#'))[j:j + 2], 16) for j in (0, 2, 4))
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=mean,
                name=channel_name.replace("_", " ").upper(),
                mode='lines',
                line=dict(color=f'rgba{rgb + (1,)}'),
            )
        )
        if show_error:
            std = channel_data.std(axis=0)
            fig.add_trace(
                go.Scatter(
                    x=np.hstack([timestamps, timestamps[::-1]]),
                    y=np.hstack([mean + std, mean - std[::-1]]),
                    name=channel_name.replace("_", " ").upper(),
                    fill='toself',
                    fillcolor=f'rgba{rgb + (0.2,)}',
                    line=dict(color=f'rgba{rgb + (0.2,)}'),
                    hoverinfo='skip',
                    showlegend=False,
                )
            )
    fig.add_vline(x=0, line_dash="dash", line_color="black", name=event_name)
    fig.update_layout(title=f"Peri-{event_name}\t({subject_str})",
                      xaxis_title="Samples",
                      yaxis_title="Amplitude (uV)",
                      showlegend=True)
    return fig


def _peri_event_image(
        data: pd.DataFrame,
        title: str,
        vlines: List[float] = None,
):
    fig = px.imshow(data, aspect='auto')
    fig.add_vline(x=0, line_width=2, line_dash="dash", line_color="black")
    for v in vlines or []:
        fig.add_vline(x=v, line_width=1, line_dash="dash", line_color="black")
    fig.update_layout(
        title=title,
        xaxis=dict(
            tickmode='array',
            tickvals=[t for t in data.columns if t % 50 == 0],
            ticktext=[t for t in data.columns if t % 50 == 0],
        ),
    ),
    return fig
