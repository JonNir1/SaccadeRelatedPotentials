import os
import re
import pickle as pkl
from typing import List, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.io as pio

import TAV.tav_helpers as tavh
from TAV.Subject import Subject

_OUTPUT_DIR = tavh.get_output_subdir(os.path.basename(__file__))
_FIGURES_DIR = os.path.join(_OUTPUT_DIR, tavh.constants.FIGURES_STR)

_EVENTS = ["saccade onset", "saccade offset"]
_EPOCH_SAMPLES_BEFORE, _EPOCH_SAMPLES_AFTER = 100, 100
_IS_RIGHTWARDS_STR = "is_rightwards"
_FOCUS_CHANNELS = ["REOG", "REOG_FILTERED"]
# _FOCUS_CHANNELS = ["REOG", "REOG_FILTERED", "F4 - F3", "F6 - F5", "F8 - F7", "FC6 - FC5", "FT8 - FT7", "T8 - T7"]

###########################


def load_or_calc(
        n_samples_before: int = _EPOCH_SAMPLES_BEFORE,
        n_samples_after: int = _EPOCH_SAMPLES_AFTER,
        enforce_trials: bool = True,
):
    os.makedirs(_OUTPUT_DIR, exist_ok=True)
    epochs, figures = {}, {}
    for i in tqdm(range(101, 111), desc="Subjects"):
        s = Subject.load_or_make(i, tavh.RESULTS_DIR)
        for evnt in tqdm(_EVENTS, desc="\tEvents"):
            subject_epochs = load_or_calc_epochs(s, evnt, n_samples_before, n_samples_after, enforce_trials)
            epochs[(s.idx, evnt)] = subject_epochs
            subject_figures = load_or_create_figures(s, evnt, subject_epochs)
            figures[(s.idx, evnt)] = subject_figures
    return epochs, figures


def load_or_calc_epochs(
        s: Subject,
        event_name: str,
        n_samples_before: int = _EPOCH_SAMPLES_BEFORE,
        n_samples_after: int = _EPOCH_SAMPLES_AFTER,
        enforce_trials: bool = True,
):
    os.makedirs(_OUTPUT_DIR, exist_ok=True)
    fname = f"s{s.idx}_{event_name.lower().replace(" ", "-")}_{tavh.constants.EPOCHS_STR.lower()}.pkl"
    try:
        return pd.read_pickle(os.path.join(_OUTPUT_DIR, fname))
    except FileNotFoundError:
        epochs = _calculate_epochs(s, event_name, n_samples_before, n_samples_after, enforce_trials)
        epochs.to_pickle(os.path.join(_OUTPUT_DIR, fname))
        return epochs


def load_or_create_figures(
        s: Subject,
        event_name: str,
        epochs: pd.DataFrame,
        show_error: bool = False,
):
    subj_figures_dir = os.path.join(_FIGURES_DIR, f"Subject_{s.idx}")
    os.makedirs(subj_figures_dir, exist_ok=True)
    figures = {}
    try:
        fname = f"{event_name.lower().replace(" ", "_")}_all_channels"
        with open(os.path.join(subj_figures_dir, fname + ".json"), "r") as f:
            figures["all_channels"] = pio.read_json(f)
    except FileNotFoundError:
        fname = f"{event_name.lower().replace(" ", "_")}_all_channels"
        all_channels = _create_all_channels_figure(epochs, f"Subject {s.idx} - {event_name.title()}", show_error)
        figures["all_channels"] = all_channels
        tavh.save_figure(all_channels, subj_figures_dir, fname)
    for channel_name in _FOCUS_CHANNELS:
        if channel_name=="REOG_FILTERED" and "onset" not in event_name:
            continue
        fname = f"{event_name.lower().replace(" ", "_")}_{channel_name}"
        try:
            with open(os.path.join(subj_figures_dir, fname + ".json"), "r") as f:
                figures[channel_name] = pio.read_json(f)
        except FileNotFoundError:
            channel_epochs = epochs.xs(channel_name, level=tavh.constants.CHANNEL_STR, drop_level=False)
            focus_channel = _create_focus_channel_figure(channel_epochs, str(s.idx), event_name)
            figures[channel_name] = focus_channel
            tavh.save_figure(focus_channel, subj_figures_dir, fname)
    return figures


def _calculate_epochs(
        s: Subject,
        event_name: str,
        n_samples_before: int = _EPOCH_SAMPLES_BEFORE,
        n_samples_after: int = _EPOCH_SAMPLES_AFTER,
        enforce_trials: bool = True,
) -> pd.DataFrame:
    event_epochs = []
    # extract epochs for all EEG & rEOG channels:
    channels = __extract_channels(s, event_name)
    event_idxs = s.get_eye_tracking_event_indices(event_name, enforce_trials)
    for channel_name, data in channels.items():
        channel_epochs = tavh.extract_epochs(data, event_idxs, n_samples_before, n_samples_after)
        channel_epochs[tavh.constants.CHANNEL_STR] = channel_name
        event_epochs.append(channel_epochs)

    # add saccade features to the epochs:
    durations = s.get_saccade_feature("duration", enforce_trials)
    amplitudes = s.get_saccade_feature("amplitude", enforce_trials)
    azimuths = s.get_saccade_feature("azimuth", enforce_trials)
    is_rightwards = (-90 <= azimuths) & (azimuths < 90)
    assert len(event_idxs) == len(azimuths) == len(durations) == len(amplitudes), "Mismatch between event indices and saccade features"
    for ch_epoch in event_epochs:
        ch_epoch[tavh.constants.DURATION_STR] = durations
        ch_epoch[tavh.constants.AMPLITUDE_STR] = amplitudes
        ch_epoch[_IS_RIGHTWARDS_STR] = is_rightwards

    # combine all channels into a single DataFrame:
    event_epochs = pd.concat(event_epochs)
    event_epochs.index.name = tavh.constants.EPOCH_STR
    event_epochs.columns.name = tavh.constants.SAMPLE_STR
    event_epochs.set_index(
        [tavh.constants.CHANNEL_STR, tavh.constants.DURATION_STR, tavh.constants.AMPLITUDE_STR, _IS_RIGHTWARDS_STR],
        append=True, inplace=True
    )
    event_epochs.sort_index(level=[tavh.constants.CHANNEL_STR, tavh.constants.DURATION_STR], inplace=True)
    return event_epochs


def _create_all_channels_figure(
        epochs: pd.DataFrame,
        title: str,
        show_error: bool = False,
) -> go.Figure:
    subtitles = ["All Saccades", "Rightwards Saccades", "Leftwards Saccades", "Direction Corrected"]
    fig = make_subplots(
        rows=3, cols=2, subplot_titles=subtitles, shared_xaxes=True, shared_yaxes=True,
        vertical_spacing=0.08, horizontal_spacing=0.05,
        specs=[[{"colspan": 2}, None], [{}, {}], [{"colspan": 2}, None]],
        x_title=tavh.constants.SAMPLES_STR.title(), y_title="Amplitude (µV)"
    )

    for i, (channel_name, channel_data) in enumerate(epochs.groupby(tavh.constants.CHANNEL_STR)):
        hex_color = tavh.constants.DEFAULT_COLORMAP[i % len(tavh.constants.DEFAULT_COLORMAP)]
        name = str(channel_name).replace("_", " ").upper()
        # traces for "All Saccades"
        fig.add_trace(row=1, col=1, trace=__create_line_trace(channel_data, name, hex_color, True))
        if show_error:
            fig.add_trace(row=1, col=1, trace=__create_area_trace(channel_data, name, hex_color))
        # traces for "Leftwards Saccades"
        is_rightwards = channel_data.index.get_level_values(_IS_RIGHTWARDS_STR)
        channel_data_left = channel_data.loc[~is_rightwards, :]
        fig.add_trace(row=2, col=2, trace=__create_line_trace(channel_data_left, name, hex_color, False))
        if show_error:
            fig.add_trace(row=2, col=2, trace=__create_area_trace(channel_data_left, name, hex_color))
        # traces for "Rightwards Saccades"
        channel_data_right = channel_data.loc[is_rightwards, :]
        fig.add_trace(row=2, col=1, trace=__create_line_trace(channel_data_right, name, hex_color, False))
        if show_error:
            fig.add_trace(row=2, col=1, trace=__create_area_trace(channel_data_right, name, hex_color))
        # traces for "Direction Corrected (L - R)"
        corrected_channel_data = channel_data.copy(deep=True)
        corrected_channel_data.loc[is_rightwards, :] = -corrected_channel_data.loc[is_rightwards, :]
        fig.add_trace(row=3, col=1, trace=__create_line_trace(corrected_channel_data, name, hex_color, False))
        if show_error:
            fig.add_trace(row=3, col=1, trace=__create_area_trace(corrected_channel_data, name, hex_color))
    fig.update_layout(
        title=title.title(),
        showlegend=True,
    )
    return fig


def _create_focus_channel_figure(
        channel_epochs: pd.DataFrame,
        subject_idx: str,
        event_name: str,
        line_color: str = "#000000",
        colorscale_name: str = "Viridis",
) -> go.Figure:
    new_channel_epochs = channel_epochs.sort_index(level=tavh.constants.DURATION_STR, inplace=False)
    is_rightwards = new_channel_epochs.index.get_level_values(_IS_RIGHTWARDS_STR)
    fig = make_subplots(
        rows=6, cols=3, shared_xaxes=False, shared_yaxes=False,
        subplot_titles=["All Saccades", "Rightwards Saccades", None, None, "Leftwards Saccades", None],
        specs=[[{"colspan": 2}, None, {}],
               [{"colspan": 2, "rowspan": 5}, None, {"rowspan": 2}],
               [None, None, None],
               [None, None, {}],
               [None, None, {"rowspan": 2}],
               [None, None, None]],
    )

    # all saccades
    fig.add_trace(
        row=1, col=1,
        trace=__create_line_trace(new_channel_epochs, "All Saccades", line_color, False)
    )
    __add_activity_heatmap(fig, new_channel_epochs, 2, 1, "onset" in event_name.lower(), colorscale_name)
    # leftwards saccades
    leftward_data = new_channel_epochs.loc[~is_rightwards, :]
    fig.add_trace(
        row=1, col=3,
        trace=__create_line_trace(leftward_data, "Leftwards Saccades", line_color, False)
    )
    __add_activity_heatmap(fig, leftward_data, 2, 3, "onset" in event_name.lower(), colorscale_name)
    # rightwards saccades
    rightward_data = new_channel_epochs.loc[is_rightwards, :]
    fig.add_trace(
        row=4, col=3,
        trace=__create_line_trace(rightward_data, "Rightwards Saccades", line_color, False)
    )
    __add_activity_heatmap(fig, rightward_data, 5, 3, "onset" in event_name.lower(), colorscale_name)

    channel_name = channel_epochs.index.get_level_values(tavh.constants.CHANNEL_STR)[0]
    title = (f"{tavh.constants.CHANNEL_STR.title()}: {subject_idx.title()} - {event_name.title()} - " +
             f"{tavh.constants.CHANNEL_STR.title()}: `{channel_name.upper()}`")
    fig.update_layout(
        title=title,
        showlegend=False,
        coloraxis_showscale=False,
        yaxis_title="Amplitude (µV)",
        yaxis3_title=tavh.constants.EPOCHS_STR.title(),
        xaxis3_title=tavh.constants.SAMPLES_STR.title(),
        xaxis6_title=tavh.constants.SAMPLES_STR.title(),
    )
    return fig


def __extract_channels(s: Subject, event_name: str) -> Dict[str, np.ndarray]:
    evnt = event_name.lower().replace(" ", "_")
    if evnt not in {"saccade_onset", "saccade_offset"}:
        raise ValueError(f"Event {event_name} not recognized")
    channels = {"REOG": s.get_channel("reog")}
    if evnt == "saccade_onset":
        channels["REOG_FILTERED"] = tavh.apply_filter(channels["REOG"], "srp")
        # TODO: add REOG_FILTERED channel for saccade offset (use different filter?)
    for channel_name in s.eeg_channels.keys():
        if channel_name.lower().endswith("z"):
            channels[channel_name] = s.get_channel(channel_name, full_ica=True)     # clean any eye-movement artifacts
            continue
        elec_num = int(re.search(r'\d+', channel_name).group())     # https://stackoverflow.com/a/7085715/8543025
        if elec_num % 2 == 1:
            # left-side channels are subtracted from right-side channels
            continue
        elec_name = channel_name[:channel_name.index(str(elec_num))]
        left_channel_name = f"{elec_name}{elec_num - 1}"
        right_channel = s.get_channel(channel_name, full_ica=True)                  # clean any eye-movement artifacts
        left_channel = s.get_channel(left_channel_name, full_ica=True)              # clean any eye-movement artifacts
        diff = right_channel - left_channel
        channels[f"{channel_name} - {left_channel_name}"] = diff
    return channels


def __create_line_trace(
        data: pd.DataFrame,
        name: str,
        hex_color: str,
        show_legend: bool,
) -> go.Scatter:
    mean = data.mean()
    rgb = tuple(int((hex_color.lstrip('#'))[j:j + 2], 16) for j in (0, 2, 4))
    trace = go.Scatter(
        x=mean.index,
        y=mean,
        name=name,
        legendgroup=name,
        showlegend=show_legend,
        line=dict(color=f'rgba{rgb + (1,)}', width=1),
    )
    return trace


def __create_area_trace(
        data: pd.DataFrame,
        name: str,
        hex_color: str,
) -> go.Scatter:
    mean = data.mean()
    std = data.std()
    rgb = tuple(int((hex_color.lstrip('#'))[j:j + 2], 16) for j in (0, 2, 4))
    trace = go.Scatter(
        x=np.hstack([mean.index, mean.index[::-1]]),
        y=np.hstack([mean + std, mean - std[::-1]]),
        name=name,
        legendgroup=name,
        showlegend=False,
        fill='toself',
        fillcolor=f'rgba{rgb + (0.2,)}',
        line=dict(color=f'rgba{rgb + (0.2,)}'),
        hoverinfo='skip',
    )
    return trace


def __add_activity_heatmap(
        fig: go.Figure,
        data: pd.DataFrame,
        row: int,
        col: int,
        positive_markers: bool = True,
        colorscale_name: str = "Viridis",
):
    fig.add_trace(
        row=row, col=col,
        trace=px.imshow(
            data.values, x=data.columns, y=np.arange(data.shape[0]), color_continuous_scale=colorscale_name.capitalize()
        ).data[0]
    )
    fig.add_vline(x=0, row=row, col=col, line_dash="dash", line_color="black")
    # add markers at for the duration of each saccade at the positive/negative side of 0
    durations = data.index.get_level_values(tavh.constants.DURATION_STR)
    if not positive_markers:
        durations = -durations
    fig.add_trace(
        row=row, col=col,
        trace=go.Scatter(
            x=[d for d in durations if abs(d) <= min(_EPOCH_SAMPLES_AFTER, _EPOCH_SAMPLES_BEFORE)],
            y=[i for i, d in enumerate(durations) if abs(d) <= min(_EPOCH_SAMPLES_AFTER, _EPOCH_SAMPLES_BEFORE)],
            mode='markers', marker=dict(size=2, color='rgba(0, 0, 0, 0.5)'),
            showlegend=False, hoverinfo='skip'
        )
    )
    fig.update_xaxes(
        row=row, col=col,
        tickmode='array',
        tickvals=[v for v in data.columns if v % 25 == 0],
        ticktext=[v for v in data.columns if v % 25 == 0],
    )
