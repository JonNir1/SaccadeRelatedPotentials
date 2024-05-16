import os
import pickle as pkl
from typing import List

import numpy as np
import pandas as pd
from scipy.stats import norm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

import TAV.tav_helpers as tavh
from TAV.Subject import Subject

_OUTPUT_DIR = tavh.get_output_subdir(os.path.basename(__file__))
_FIGURES_DIR = os.path.join(_OUTPUT_DIR, tavh.FIGURES_STR)

WINDOW_SIZES = np.arange(21)
COLORS = {
    "HR": "#2166ac", "FAR": "#8c510a", "d'": "#762a83",
    "c": "#b35806", "PPV": "#01665e", "F1": "#c51b7d",  # comment out if not needed
}

#####################


def load_or_calc_saccade_onset():
    os.makedirs(_FIGURES_DIR, exist_ok=True)
    event_name = "saccade_onset"
    filename_format = "%d_" + event_name
    measures, subject_figures = {}, {}
    for idx in range(101, 111):
        # load subject stats
        stats_file_path = os.path.join(_OUTPUT_DIR, filename_format % idx + ".pkl")
        try:
            with open(stats_file_path, "rb") as f:
                measures[idx] = pkl.load(f)
        except FileNotFoundError:
            s = Subject.load_or_make(idx, tavh.OUTPUT_DIR)
            measures[idx] = saccade_onset_detection_measures(s, WINDOW_SIZES)
            with open(stats_file_path, "wb") as f:
                pkl.dump(measures[idx], f)
        # load subject fig
        figure_file_path = os.path.join(_FIGURES_DIR, filename_format % idx + ".json")
        try:
            with open(figure_file_path, "rb") as f:
                subject_figures[idx] = pio.read_json(f)
        except FileNotFoundError:
            subject_figures[idx] = _create_subject_measures_figure(measures[idx], idx, event_name)
            tavh.save_figure(subject_figures[idx], _FIGURES_DIR, filename_format % idx)
    try:
        with open(os.path.join(_FIGURES_DIR, "mean_fig.json"), 'rb') as f:
            mean_fig = pio.read_json(f)
    except FileNotFoundError:
        mean_fig = _create_mean_measures_figure(list(measures.values()))
        tavh.save_figure(mean_fig, _FIGURES_DIR, "mean_fig")
    return measures, subject_figures, mean_fig


def saccade_onset_detection_measures(s: Subject, window_sizes: np.ndarray):
    """
    Calculates Signal Detection Theory measures for Saccade Onset detection using Eye-Tracking and REOG data.
    Returns a DataFrame with the calculated measures for each of the specified window sizes.
    Calculated measures include:
        - P: number of positive events in the Ground-Truth
        - N: number of negative events in the Ground-Truth
        - PP: number of positive events in the Predictions
        - TP: number of correctly detected positive events
        - HR: Hit Rate (Sensitivity, Recall, True Positive Rate)
        - FAR: False Alarm Rate (False Positive Rate, Type I Error)
        - PPV: Positive Predictive Value (Precision)
        - F1: F1 Score
        - d': d-prime, Sensitivity Index
        - beta: Response Bias
        - c: Decision Criterion
    """
    num_trial_idxs = sum(s.get_is_trial_channel())
    et_onset_idxs = s.get_eye_tracking_event_indices('saccade_onset', False)
    et_onset_channel = s.create_boolean_event_channel(et_onset_idxs, enforce_trials=True)
    reog_onset_idxs = s.calculate_reog_saccade_onset_indices(filter_name='srp', snr=3.5, enforce_trials=False)
    reog_onset_channel = s.create_boolean_event_channel(reog_onset_idxs, enforce_trials=True)
    P = et_onset_channel.sum()      # number of GT positive events
    PP = reog_onset_channel.sum()   # number of Predicted positive events
    all_matched_idxs = _match_events(et_onset_channel, reog_onset_channel)
    measures = {}
    for ws in window_sizes:
        double_window = 2 * ws + 1  # window size on both sides of the event
        N = (num_trial_idxs - double_window * P) / double_window  # number of windows with no GT events
        matched_idxs = all_matched_idxs[abs(np.diff(all_matched_idxs, axis=1).flatten()) <= ws]
        TP = len(matched_idxs)
        window_measures = _calculate_signal_detection_measures(P, N, PP, TP)
        window_measures.name = ws
        measures[ws] = window_measures
    results = pd.DataFrame(measures).T
    results.index.name = 'Window Size'
    results.columns.name = 'Measure'
    return results


def _calculate_signal_detection_measures(P: int, N: int, PP: int, TP: int) -> pd.Series:
    """
    Calculates Signal Detection Theory measures while correcting for floor/ceiling effects (to avoid division by zero).
    :param P: number of positive events in the Ground-Truth
    :param N: number of negative events in the Ground-Truth
    :param PP: number of positive events in the Predictions
    :param TP: number of correctly detected positive events
    :return: Series containing the calculated measures:
        - P: number of positive events in the Ground-Truth
        - N: number of negative events in the Ground-Truth
        - PP: number of positive events in the Predictions
        - TP: number of correctly detected positive events
        - HR: Hit Rate (Sensitivity, Recall, True Positive Rate)
        - FAR: False Alarm Rate (False Positive Rate, Type I Error)
        - PPV: Positive Predictive Value (Precision)
        - F1: F1 Score
        - d': d-prime, Sensitivity Index
        - beta: Response Bias
        - c: Decision Criterion
    """
    assert P >= 0 and N >= 0 and PP >= 0 and TP >= 0, "All values must be non-negative"
    assert TP <= P and TP <= PP, "TP must be less than or equal to P and PP"
    hit_rate = __calc_rate(P, TP)        # aka sensitivity, recall, true positive rate (TPR)
    fa_rate = __calc_rate(N, PP - TP)    # aka false alarm rate, false positive rate (FPR), type I error
    precision = __calc_rate(PP, TP)      # aka positive predictive value (PPV)
    f1_score = 2 * precision * hit_rate / (precision + hit_rate) if precision + hit_rate > 0 else 0
    d_prime, beta, criterion = __calc_dprime_beta_criterion(hit_rate, fa_rate)
    return pd.Series({
        'P': P, 'N': N, 'PP': PP, 'TP': TP, 'HR': hit_rate, 'FAR': fa_rate,
        'PPV': precision, 'F1': f1_score, 'd\'': d_prime, 'beta': beta, 'c': criterion,
    })


def _create_subject_measures_figure(measures: pd.DataFrame, subj_id: int, event: str) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,)
    ws = measures.index
    for column, color in COLORS.items():
        row = 2 if column in {"d'", "c"} else 1
        fig.add_trace(
            row=row, col=1,
            trace=go.Scatter(
                x=ws, y=measures[column], mode='lines+markers', name=column, marker=dict(color=color)
            ),
        )
    fig.update_layout(
        title=f"Subject {subj_id} - {event.replace("_", " ").title()}",
        xaxis=dict(title="Window Size (ms)"),
        showlegend=True
    )
    return fig


def _create_mean_measures_figure(stats: List[pd.DataFrame]) -> go.Figure:
    window_sizes, col_names = stats[0].index, stats[0].columns.to_list()
    mean_fig = make_subplots(rows=2, cols=1, shared_xaxes=True,)
    for column, color in COLORS.items():
        row = 2 if column in {"d'", "c"} else 1
        mean_fig.add_trace(
            row=row, col=1,
            trace=go.Scatter(
                x=window_sizes,
                y=np.mean([s[column] for s in stats], axis=0),
                error_y=dict(type='data', array=np.std([s[column] for s in stats], axis=0)),
                mode='lines+markers',
                name=column,
                marker=dict(color=color)
            )
        )
        for stat in stats:
            # add thin lines for individual subjects
            mean_fig.add_trace(
                row=row, col=1,
                trace=go.Scatter(
                    x=window_sizes,
                    y=stat[column],
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


def _match_events(is_gt: np.ndarray, is_pred: np.ndarray) -> np.ndarray:
    """
    Matches between Ground-Truth and Predicted events, such that there is minimal difference between the indices of
    the matched events. Returns a 2D array where each row is a matching pair of indices, where the first column is the
    GT index and the second column is the Predicted index.
    See also here: https://stackoverflow.com/q/78484847/8543025

    :param is_gt: boolean array indicating the occurrence of Ground-Truth events
    :param is_pred: boolean array indicating the occurrence of Predicted events
    :return: m×2 array of matched indices (0 <= m <= min(sum(is_gt), sum(is_pred))
    """
    assert is_gt.shape == is_pred.shape, "Ground-Truth and Predicted arrays must have the same shape"
    gt_idxs = np.where(is_gt)[0]
    pred_idxs = np.where(is_pred)[0]
    diffs = abs(gt_idxs[:, None] - pred_idxs[None, :])
    rowwise_argmin = np.stack([diffs.argmin(0), np.arange(diffs.shape[1])]).T
    colwise_argmin = np.stack([np.arange(diffs.shape[0]), diffs.argmin(1)]).T
    is_matching = (rowwise_argmin[:, None] == colwise_argmin).all(-1).any(1)
    idxs = rowwise_argmin[is_matching]
    matching_indices = np.stack([gt_idxs[idxs[:, 0]], pred_idxs[idxs[:, 1]]]).T
    return matching_indices


def __calc_rate(true_count: int, detected_count: int) -> float:
    """
    Calculates the Hit Rate / False Alarm Rate while adjusting for floor/ceiling effects.
    See https://lindeloev.net/calculating-d-in-python-and-php/ for more details.
    """
    assert 0 <= detected_count <= true_count, "Detected Count must be between 0 and True Count"
    quarter_true = 0.25 / true_count
    rate = detected_count / true_count if true_count > 0 else 0
    if rate == 0:
        rate = quarter_true
    if rate == 1:
        rate = 1 - quarter_true
    return rate


def __calc_dprime_beta_criterion(hr: float, far: float) -> (float, float, float):
    """
    Calculates d-prime beta and criterion from Hit Rate and False Alarm Rate, while adjusting for floor/ceiling effects.
    See https://lindeloev.net/calculating-d-in-python-and-php/ for more details.
    """
    assert 0 <= hr <= 1, "Hit Rate must be between 0 and 1"
    assert 0 <= far <= 1, "False Alarm Rate must be between 0 and 1"
    Z = norm.ppf
    d_prime = Z(hr) - Z(far)
    beta = np.exp((Z(far)**2 - Z(hr)**2) / 2)
    criterion = -0.5 * (Z(hr) + Z(far))
    return d_prime, beta, criterion
