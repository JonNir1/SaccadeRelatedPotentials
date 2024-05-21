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
_FIGURES_DIR = os.path.join(_OUTPUT_DIR, tavh.FIGURES_STR)

_FOCUS_CHANNELS = ["REOG", "REOG_FILTERED", "F4 - F3", "F6 - F5", "F8 - F7", "FC6 - FC5", "FT8 - FT7", "T8 - T7"]
_EPOCH_SAMPLES_BEFORE, _EPOCH_SAMPLES_AFTER = 250, 250
_DEFAULT_COLORMAP = px.colors.qualitative.Dark24


def load_or_calc(calc_mean: bool = True):
    os.makedirs(_FIGURES_DIR, exist_ok=True)
    subject_epochs, subject_is_rightwards = {}, {}
    subject_figures = {}
    for i in tqdm(range(101, 111), desc="Subjects"):
        s = Subject.load_or_make(i, tavh.OUTPUT_DIR)
        subject_epochs[s.idx] = load_or_calc_epochs(s)
        azimuth = s.get_saccade_feature("azimuth", enforce_trials=True)
        subject_is_rightwards[s.idx] = (-90 <= azimuth) & (azimuth < 90)
        subject_figures[s.idx] = load_or_create_subject_figures(
            s.idx, subject_epochs[s.idx], subject_is_rightwards[s.idx]
        )
    if not calc_mean:
        return subject_epochs, subject_figures, None
    mean_event_figures = create_mean_subject_event_figures(subject_epochs, subject_is_rightwards)
    mean_channel_figures = create_mean_subject_channel_figures(subject_epochs, subject_is_rightwards)
    mean_figures = {
        event_name: {"all_channels": mean_event_figures[event_name], **mean_channel_figures[event_name]}
        for event_name in mean_event_figures.keys()
    }
    return subject_epochs, subject_figures, mean_figures


def load_or_calc_epochs(s: Subject) -> pd.DataFrame:
    os.makedirs(_OUTPUT_DIR, exist_ok=True)
    fname = f"s{s.idx}_{tavh.EPOCHS_STR.lower()}"
    try:
        with open(os.path.join(_OUTPUT_DIR, f"{fname}.pkl"), "rb") as f:
            return pkl.load(f)
    except FileNotFoundError:
        epochs = {}
        for event_name in {"saccade onset", "saccade offset"}:
            epochs[event_name] = _calculate_event_epochs(
                s, event_name, _EPOCH_SAMPLES_BEFORE, _EPOCH_SAMPLES_AFTER, True
            )
        # store as DataFrame
        epochs = pd.DataFrame(epochs)
        epochs.index.name = tavh.CHANNELS_STR
        epochs.columns.name = tavh.EVENTS_STR
        with open(os.path.join(_OUTPUT_DIR, f"{fname}.pkl"), "wb") as f:
            pkl.dump(epochs, f)
        return epochs


def load_or_create_subject_figures(
        subj_idx: int,
        epochs: pd.DataFrame,
        is_rightward: np.ndarray,
) -> Dict[str, go.Figure]:
    subj_dir = os.path.join(_FIGURES_DIR, f"s{subj_idx}")
    os.makedirs(subj_dir, exist_ok=True)
    figures = {}
    for event_name in epochs.columns:
        try:
            with open(os.path.join(subj_dir, f"{event_name}_all_channels.json"), "rb") as f:
                figures[event_name] = pio.read_json(f)
        except FileNotFoundError:
            fig = _create_event_figure(epochs[event_name], is_rightward, f"Subject {subj_idx} - {event_name}")
            tavh.save_figure(fig, subj_dir, f"{event_name}_all_channels")
            figures[event_name] = fig

        for focus_channel in _FOCUS_CHANNELS:
            try:
                with open(os.path.join(subj_dir, f"{event_name}_{focus_channel}.json"), "rb") as f:
                    figures[f"{event_name}_{focus_channel}"] = pio.read_json(f)
            except FileNotFoundError:
                channel_data = epochs.loc[focus_channel, event_name]
                if channel_data is None or np.isnan(channel_data).all().all() or channel_data.empty:
                    continue
                fig = _create_channel_figure(
                    channel_data, is_rightward, focus_channel, f"Subject {subj_idx} - {event_name} - {focus_channel}"
                )
                tavh.save_figure(fig, subj_dir, f"{event_name}_{focus_channel}")
                figures[f"{event_name}_{focus_channel}"] = fig
    return figures


def create_mean_subject_event_figures(
        subject_epochs: Dict[int, pd.DataFrame],
        subject_is_rightwards: Dict[int, np.ndarray],
        colormap: List[str] = None,
        show_error: bool = False,
) -> Dict[str, go.Figure]:
    mean_subj_dir = os.path.join(_FIGURES_DIR, "mean subject")
    os.makedirs(mean_subj_dir, exist_ok=True)
    colormap = colormap or tavh.DEFAULT_COLORMAP
    mean_epochs = _aggregate_mean_epochs(subject_epochs, subject_is_rightwards)
    figures = {}
    for event_name in tqdm(mean_epochs.columns, desc="Mean Subject - Event Figures"):
        if event_name.endswith("_corrected"):
            # skip corrected epochs - they are already included in figures of raw epochs
            continue
        event_fig = make_subplots(
            rows=2, cols=1, shared_xaxes='all', shared_yaxes='all', x_title=tavh.SAMPLES_STR, y_title="Amplitude (µV)",
            subplot_titles=["Raw", "Direction Corrected"], horizontal_spacing=0.05,
        )
        for n, channel_name in enumerate(tqdm(mean_epochs.index, desc="\tChannels")):
            # top row: raw epochs
            raw_data = mean_epochs.loc[channel_name, event_name]
            if raw_data is None or np.isnan(raw_data).all().all() or raw_data.empty:
                continue
            event_fig.add_trace(
                row=1, col=1,
                trace=__create_line_trace(raw_data, colormap[n % len(colormap)], channel_name, showlegend=True),
            )
            if show_error:
                event_fig.add_trace(
                    row=1, col=1,
                    trace=__create_error_area(raw_data, colormap[n % len(colormap)], channel_name),
                )
            event_fig.add_vline(row=1, col=1, x=0, line=dict(color='black', width=1, dash='dash'))
            # bottom row: direction-corrected epochs
            corrected_data = mean_epochs.loc[channel_name, f"{event_name}_corrected"]
            event_fig.add_trace(
                row=2, col=1,
                trace=__create_line_trace(corrected_data, colormap[n % len(colormap)], channel_name, showlegend=False),
            )
            if show_error:
                event_fig.add_trace(
                    row=2, col=1,
                    trace=__create_error_area(corrected_data, colormap[n % len(colormap)], channel_name),
                )
            event_fig.add_vline(row=2, col=1, x=0, line=dict(color='black', width=1, dash='dash'))
        event_fig.update_layout(
            title_text=f"Mean Subject - {event_name.replace('_', ' ').title()}",
            showlegend=True
        )
        tavh.save_figure(event_fig, mean_subj_dir, f"{event_name}_all_channels")
        figures[event_name] = event_fig
    return figures


def create_mean_subject_channel_figures(
        subject_epochs: Dict[int, pd.DataFrame],
        subject_is_rightwards: Dict[int, np.ndarray],
        line_color: str = tavh.DEFAULT_COLORMAP[0],
        scale_color: str = "Viridis",
        show_error: bool = False,
) -> Dict[str, Dict[str, go.Figure]]:
    mean_subj_dir = os.path.join(_FIGURES_DIR, "mean subject")
    os.makedirs(mean_subj_dir, exist_ok=True)
    mean_epochs = _aggregate_mean_epochs(subject_epochs, subject_is_rightwards)
    figures = {}
    for event_name in mean_epochs.columns:
        if event_name.endswith("_corrected"):
            # skip corrected epochs - they are already included in figures of raw epochs
            continue
        figures[event_name] = {}
        for channel_name in tqdm(_FOCUS_CHANNELS, desc=f"Mean Subject - {event_name.title()} - Channel Figures"):
            raw_channel_data = mean_epochs.loc[channel_name, event_name]
            if raw_channel_data is None or np.isnan(raw_channel_data).all().all() or raw_channel_data.empty:
                continue
            channel_fig = make_subplots(
                rows=2, cols=2, shared_xaxes='all', shared_yaxes='rows', x_title=tavh.SAMPLES_STR,
                row_heights=[0.25, 0.75], horizontal_spacing=0.05, vertical_spacing=0.05,
                column_titles=["Raw", "Direction Corrected"],
            )
            # line traces on top row - raw data (left)
            channel_fig.add_trace(
                row=1, col=1,
                trace=__create_line_trace(raw_channel_data, line_color, channel_name, showlegend=False),
            )
            if show_error:
                channel_fig.add_trace(
                    row=1, col=1,
                    trace=__create_error_area(raw_channel_data, line_color, channel_name),
                )
            # line traces on top row - corrected data (right)
            corrected_channel_data = mean_epochs.loc[channel_name, f"{event_name}_corrected"]
            channel_fig.add_trace(
                row=1, col=2,
                trace=__create_line_trace(corrected_channel_data, line_color, channel_name, showlegend=False),
            )
            if show_error:
                channel_fig.add_trace(
                    row=1, col=2,
                    trace=__create_error_area(corrected_channel_data, line_color, channel_name),
                )
            # heatmaps on bottom row - raw data (left) and corrected data (right)
            channel_fig.add_trace(
                row=2, col=1, trace=px.imshow(
                    raw_channel_data, aspect='auto', color_continuous_scale=scale_color
                ).data[0],
            )
            channel_fig.add_trace(
                row=2, col=2, trace=px.imshow(
                    corrected_channel_data, aspect='auto', color_continuous_scale=scale_color
                ).data[0],
            )
            # add vline @ 0 for each subplot
            [channel_fig.add_vline(row=r + 1, col=c + 1, x=0, line=dict(color='black', width=1, dash='dash'))
             for r in range(2) for c in range(2)]
            channel_fig.update_layout(
                title_text=f"Mean Subject - {event_name.replace('_', ' ').title()} - {channel_name}",
                showlegend=False,
                coloraxis_showscale=False,
                yaxis_title="Amplitude (µV)",
                yaxis3_title=tavh.EPOCHS_STR.title(),
            )
            # store figure
            tavh.save_figure(channel_fig, mean_subj_dir, f"{event_name}_{channel_name}")
            figures[event_name][channel_name] = channel_fig
    return figures


def _calculate_event_epochs(
        s: Subject,
        event_name: str,
        n_samples_before: int = _EPOCH_SAMPLES_BEFORE,
        n_samples_after: int = _EPOCH_SAMPLES_AFTER,
        enforce_trials: bool = True,
) -> pd.Series:
    event_idxs = s.get_eye_tracking_event_indices(event_name, enforce_trials)
    channels = __extract_channels(s, event_name)
    event_epochs = {}
    for channel_name, data in channels.items():
        channel_epochs = tavh.extract_epochs(data, event_idxs, n_samples_before, n_samples_after)
        event_epochs[channel_name] = channel_epochs
    event_epochs = pd.Series(event_epochs, name=event_name.replace(" ", "_").lower())
    event_epochs.index.name = tavh.CHANNELS_STR
    return event_epochs


def _aggregate_mean_epochs(
        subject_epochs: Dict[int, pd.DataFrame],
        subject_is_rightwards: Dict[int, np.ndarray],
) -> pd.DataFrame:
    mean_dir = os.path.join(_FIGURES_DIR, "mean subject")
    os.makedirs(mean_dir, exist_ok=True)
    events, channels = subject_epochs[101].columns, subject_epochs[101].index  # example subject
    mean_epochs = {}
    for evnt in events:
        raw_event_epochs, corrected_event_epochs = {}, {}
        for ch_name in channels:
            raw_ch_epochs, corrected_ch_epochs = [], []
            for subj_idx in subject_epochs.keys():
                subj_chan_epoch = subject_epochs[subj_idx].loc[ch_name, evnt]
                if subj_chan_epoch is None or np.isnan(subj_chan_epoch).all().all() or subj_chan_epoch.empty:
                    continue
                raw_ch_epochs.append(subj_chan_epoch)
                subj_is_rightward = subject_is_rightwards[subj_idx]
                subj_chan_epoch_corrected = subj_chan_epoch.copy(deep=True)
                subj_chan_epoch_corrected[subj_is_rightward] = -subj_chan_epoch_corrected[subj_is_rightward]
                corrected_ch_epochs.append(subj_chan_epoch_corrected)
            if not raw_ch_epochs or not corrected_ch_epochs:
                continue
            raw_event_epochs[ch_name] = pd.concat(raw_ch_epochs).groupby(level=0).mean()
            corrected_event_epochs[ch_name] = pd.concat(corrected_ch_epochs).groupby(level=0).mean()
        raw_event_epochs = pd.Series(raw_event_epochs, name=evnt.replace(" ", "_").lower())
        corrected_event_epochs = pd.Series(corrected_event_epochs, name=f"{evnt.replace(' ', '_').lower()}_corrected")
        mean_epochs[raw_event_epochs.name] = raw_event_epochs
        mean_epochs[corrected_event_epochs.name] = corrected_event_epochs
    mean_epochs = pd.DataFrame(mean_epochs)
    mean_epochs.index.name = tavh.CHANNELS_STR
    mean_epochs.columns.name = tavh.EVENTS_STR
    # save to file  # commented out to avoid overwriting
    # with open(os.path.join(mean_dir, "mean_epochs.pkl"), "wb") as f:
    #     pkl.dump(mean_epochs, f)
    return mean_epochs


def _create_event_figure(
        epochs: pd.Series,
        is_rightward: np.ndarray,
        title: str,
        colormap: List[str] = None,
        show_error: bool = False,
) -> go.Figure:
    colormap = colormap or tavh.DEFAULT_COLORMAP
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes='all', shared_yaxes='all', x_title=tavh.SAMPLES_STR, y_title="Amplitude (µV)",
        subplot_titles=["Raw", "Direction Corrected"], horizontal_spacing=0.05,
    )
    for n, channel_name in enumerate(epochs.index):
        data = epochs.loc[channel_name]
        if data is None or np.isnan(data).all().all() or data.empty:
            continue
        data = data.copy(deep=True)     # avoid modifying original data
        for r in range(2):
            if r == 1:
                # correct for saccade direction
                data[is_rightward] = -data[is_rightward]
            fig.add_trace(
                row=r + 1, col=1,
                trace=__create_line_trace(data, colormap[n % len(colormap)], channel_name, showlegend=r == 1),
            )
            if show_error:
                fig.add_trace(
                    row=r + 1, col=1,
                    trace=__create_error_area(data, colormap[n % len(colormap)], channel_name),
                )
            fig.add_vline(row=r + 1, col=1, x=0, line=dict(color='black', width=1, dash='dash'))
    fig.update_layout(
        title_text=title,
        showlegend=True
    )
    return fig


def _create_channel_figure(
        channel_data: pd.DataFrame,
        is_rightward: np.ndarray,
        channel_name: str,
        title: str,
        line_color: str = tavh.DEFAULT_COLORMAP[0],
        scale_color: str = "Viridis",
        show_error: bool = False,
) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=2, shared_xaxes='all', shared_yaxes='rows', x_title=tavh.SAMPLES_STR, row_heights=[0.25, 0.75],
        horizontal_spacing=0.05, vertical_spacing=0.05, column_titles=["Raw", "Direction Corrected"],
    )
    corrected_data = channel_data.copy(deep=True)
    corrected_data[is_rightward] = -corrected_data[is_rightward]
    # line traces on top row
    fig.add_trace(
        row=1, col=1,
        trace=__create_line_trace(channel_data, line_color, channel_name, showlegend=False),
    )
    if show_error:
        fig.add_trace(
            row=1, col=1,
            trace=__create_error_area(channel_data, line_color, channel_name),
        )
    fig.add_trace(
        row=1, col=2,
        trace=__create_line_trace(corrected_data, line_color, channel_name, showlegend=False),
    )
    if show_error:
        fig.add_trace(
            row=1, col=2,
            trace=__create_error_area(corrected_data, line_color, channel_name),
        )
    # heatmaps on bottom row
    fig.add_trace(
        row=2, col=1, trace=px.imshow(channel_data, aspect='auto', color_continuous_scale=scale_color).data[0],
    )
    fig.add_trace(
        row=2, col=2, trace=px.imshow(corrected_data, aspect='auto', color_continuous_scale=scale_color).data[0],
    )
    # add vline @ 0 for each subplot
    [fig.add_vline(row=r + 1, col=c + 1, x=0, line=dict(color='black', width=1, dash='dash'))
     for r in range(2) for c in range(2)]
    fig.update_layout(
        title_text=title,
        showlegend=False,
        coloraxis_showscale=False,
        yaxis_title="Amplitude (µV)",
        yaxis3_title=tavh.EPOCHS_STR.title(),
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



def __create_line_trace(data: pd.DataFrame, color: str, name: str, showlegend: bool) -> go.Scatter:
    ts = data.columns
    mean = data.mean(axis=0)
    rgb = tuple(int((color.lstrip('#'))[j:j + 2], 16) for j in (0, 2, 4))
    name = name.replace("_", " ").upper()
    trace = go.Scatter(
        x=ts,
        y=mean,
        name=name,
        legendgroup=name,
        showlegend=showlegend,
        mode='lines',
        line=dict(color=f'rgba{rgb + (1,)}', width=1),
    )
    return trace


def __create_error_area(data: pd.DataFrame, color: str, name: str) -> go.Scatter:
    ts = data.columns
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    rgb = tuple(int((color.lstrip('#'))[j:j + 2], 16) for j in (0, 2, 4))
    name = name.replace("_", " ").upper()
    trace = go.Scatter(
        x=np.hstack([ts, ts[::-1]]),
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
