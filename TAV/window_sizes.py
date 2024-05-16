import os
from typing import List

import numpy as np
import pandas as pd
import pickle as pkl
from tqdm import tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

import TAV.tav_helpers as tavh
from TAV.Subject import Subject

_OUTPUT_DIR = tavh.get_output_subdir(os.path.basename(__file__))
_FIGURES_DIR = os.path.join(_OUTPUT_DIR, tavh.FIGURES_STR)

COLUMNS = {'HR': "#1f78b4", 'PPV': "#2ca25f", 'FR': "#e41a1c", "Tav FR": "#fc8d62"}
WINDOW_SIZES = np.arange(1, 21)

#####################


def _true_positive_count(gt_idxs, pred_idxs, half_window: int = 8, trial_idxs: np.ndarray = None) -> int:
    """
    Calculates the number of positive events that are correctly predicted by the model, meaning that the prediction
    occurs within a window of size `2 * half_window + 1` centered around the GT event.
    If `trial_idxs` is provided, only events (GT & Pred) that occur during a trial are considered.

    :param gt_idxs: sample indices where the Ground-Truth events occur
    :param pred_idxs: sample indices where the Predicted events occur
    :param half_window: half the size of the window around the prediction
    :param trial_idxs: sample indices where the trials occur
    :return: tpr: float in range [0, 1]
    """
    gt_idxs, pred_idxs = __verify_input(gt_idxs, pred_idxs, half_window, trial_idxs)
    windows_around_gt = np.array([np.arange(i - half_window, i + half_window + 1) for i in gt_idxs])
    hit_count = np.isin(windows_around_gt, pred_idxs).any(axis=1).sum()
    return hit_count


def _true_positive_rate(gt_idxs, pred_idxs, half_window: int = 8, trial_idxs: np.ndarray = None):
    """ Calculates TP/P: the proportion of correct positive predictions out of all Ground-Truth positive events. """
    hit_count = _true_positive_count(gt_idxs, pred_idxs, half_window, trial_idxs)
    return hit_count / len(gt_idxs)


def _positive_predictive_value(gt_idxs, pred_idxs, half_window: int = 8, trial_idxs: np.ndarray = None):
    """ Calculates TP/PP: the proportion of correct positive predictions out of all positive predictions. """
    hit_count = _true_positive_count(pred_idxs, gt_idxs, half_window, trial_idxs)
    return hit_count / len(pred_idxs)


def _false_alarm_rate(gt_idxs, pred_idxs, num_samples: int, half_window: int = 8, trial_idxs: np.ndarray = None):
    """
    Calculates the false alarm rate between Ground-Truth and Predictions, where a `false alarm` is defined as a
    prediction that does not have a corresponding GT within a window of size `2 * half_window + 1` centered around the
    prediction. If `trial_idxs` is provided, only events (GT & Pred) that occur during a trial are considered.
    The number of false alarms is divided by the number of GT windows that do not contain an event.
    """
    gt_idxs, pred_idxs = __verify_input(gt_idxs, pred_idxs, half_window, trial_idxs)
    is_gt_event_window = __is_event_window(num_samples, gt_idxs, half_window)
    is_pred_event_window = __is_event_window(num_samples, pred_idxs, half_window)
    is_pred_not_gt_window = is_pred_event_window & ~is_gt_event_window
    if trial_idxs is not None:
        # only consider events that occur during a trial
        is_trial_window = __is_event_window(num_samples, trial_idxs, half_window)
        is_gt_event_window = is_gt_event_window[is_trial_window]
        is_pred_not_gt_window = is_pred_not_gt_window[is_trial_window]
    return np.sum(is_pred_not_gt_window) / np.sum(~is_gt_event_window)


def __verify_input(gt_idxs, pred_idxs, half_window: int = 8, trial_idxs: np.ndarray = None):
    assert half_window >= 0, "Half-window size must be non-negative"
    gt_idxs_copy = gt_idxs.copy()
    pred_idxs_copy = pred_idxs.copy()
    if trial_idxs is not None:
        gt_idxs_copy = gt_idxs_copy[np.isin(gt_idxs_copy, trial_idxs)]
        pred_idxs_copy = pred_idxs_copy[np.isin(pred_idxs_copy, trial_idxs)]
    return gt_idxs_copy, pred_idxs_copy


def __is_event_window(num_samples: int, is_event_idxs: np.ndarray, half_window: int = 8) -> np.ndarray:
    """
    Given an array of indices where events occur, returns a boolean array where True values indicate the presence
    of an event within a window of size `2 * half_window + 1` centered around the event.
    The output array has size of:
        is_event_idxs.size // (2 * half_window + 1) or 1+(is_event_idxs.size // (2 * half_window + 1))
        (if is_event_idxs.size % (2 * half_window + 1) != 0, in which case the last window is smaller than the rest)
    """
    window_size = 2 * half_window + 1
    is_event_array = tavh.create_boolean_array(num_samples, is_event_idxs)
    is_event_windows = np.split(is_event_array, np.arange(window_size, is_event_array.size, window_size))
    return np.array(list(map(any, is_event_windows)))

###########################


def _measure_detection_performance(s: Subject, window_sizes: np.ndarray) -> pd.DataFrame:
    saccade_onset_idxs = s.get_eye_tracking_event_indices('saccade_onset', True)
    is_reog_saccade_onset_idxs = s.calculate_reog_saccade_onset_indices(filter_name='srp', snr=3.5, enforce_trials=True)
    stats = np.zeros((len(window_sizes), len(COLUMNS)))
    for j, ws in tqdm(enumerate(window_sizes), desc="\tWindow Sizes", leave=False):
        tpr = _true_positive_rate(saccade_onset_idxs, is_reog_saccade_onset_idxs, half_window=ws)
        ppv = _positive_predictive_value(saccade_onset_idxs, is_reog_saccade_onset_idxs, half_window=ws)
        fr = _false_alarm_rate(saccade_onset_idxs, is_reog_saccade_onset_idxs, s.num_samples, half_window=ws,
                               trial_idxs=np.where(s.get_is_trial_channel())[0])
        tav_fr = _false_alarm_rate(saccade_onset_idxs, is_reog_saccade_onset_idxs, s.num_samples, half_window=ws)
        stats[j] = [tpr, ppv, fr, tav_fr]
    stats = pd.DataFrame(stats, columns=list(COLUMNS.keys()))
    return stats


def _create_subject_perf_figure(stats: pd.DataFrame, subj_id: int) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("Hit Rate", "False Alarm Rate"))
    for i, col in enumerate(stats.columns):
        color = COLUMNS[col]
        fig.add_trace(
            col=1, row=i // 2 + 1,
            trace=go.Scatter(
                x=stats.index,
                y=stats[col],
                mode='lines+markers',
                name=col,
                marker=dict(color=color)
            )
        )
    fig.update_layout(title=f"Subject {subj_id}",
                      xaxis_title="Window Size (ms)",
                      showlegend=True)
    return fig


def _create_mean_performance_figure(stats: List[pd.DataFrame]) -> go.Figure:
    window_sizes, col_names = stats[0].index, stats[0].columns.to_list()
    mean_fig = make_subplots(rows=2, cols=1,
                             shared_xaxes=True,
                             subplot_titles=("Mean Hit Rate", "Mean False Alarm Rate"))
    for i, col in enumerate(col_names):
        color = COLUMNS[col]
        mean_fig.add_trace(
            col=1, row=i//2 + 1,
            trace=go.Scatter(
                x=window_sizes,
                y=np.mean([s[col] for s in stats], axis=0),
                error_y=dict(type='data', array=np.std([s[col] for s in stats], axis=0)),
                mode='lines+markers',
                name=col,
                marker=dict(color=color)
            )
        )
        for j, stat in enumerate(stats):
            mean_fig.add_trace(
                col=1, row=i//2 + 1,
                trace=go.Scatter(
                    x=window_sizes,
                    y=stat[col],
                    mode='lines',
                    showlegend=False,
                    line=dict(color=color, width=0.2)
                )
            )
    mean_fig.update_layout(
        title="Mean Performance",
        xaxis_title="Window Size (ms)",
        showlegend=True
    )
    return mean_fig

###########################


def load_or_calc():
    os.makedirs(_FIGURES_DIR, exist_ok=True)
    try:
        with open(os.path.join(_OUTPUT_DIR, "subject_stats.pkl"), 'rb') as f:
            subject_stats = pkl.load(f)
    except FileNotFoundError:
        subject_stats = {}
        for i in tqdm(range(101, 111), desc="Subjects"):
            s = Subject.load_or_make(i)
            stats = _measure_detection_performance(s, WINDOW_SIZES)
            subject_stats[s.idx] = stats
        with open(os.path.join(_OUTPUT_DIR, "subject_stats.pkl"), 'wb') as f:
            pkl.dump(subject_stats, f)
    try:
        with open(os.path.join(_FIGURES_DIR, "subject_figs.pkl"), 'rb') as f:
            subject_figs = pkl.load(f)
    except FileNotFoundError:
        subject_figs = {}
        for idx, stats in tqdm(subject_stats.items(), desc="Subject Figures"):
            fig = _create_subject_perf_figure(stats, idx)
            tavh.save_figure(fig, _FIGURES_DIR, f"subject_{idx}_statistics")
            subject_figs[idx] = fig
        with open(os.path.join(_FIGURES_DIR, "subject_figs.pkl"), 'wb') as f:
            pkl.dump(subject_figs, f)
    try:
        with open(os.path.join(_FIGURES_DIR, "mean_fig.json"), 'rb') as f:
            mean_fig = pio.read_json(f)
    except FileNotFoundError:
        mean_fig = _create_mean_performance_figure(list(subject_stats.values()))
        tavh.save_figure(mean_fig, _FIGURES_DIR, "mean_fig")
    return subject_stats, subject_figs, mean_fig
