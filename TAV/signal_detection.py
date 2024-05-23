import os
import pickle as pkl
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm
from tqdm import tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

import TAV.tav_helpers as tavh
from TAV.Subject import Subject

_OUTPUT_DIR = tavh.get_output_subdir(os.path.basename(__file__))
_FIGURES_DIR = os.path.join(_OUTPUT_DIR, tavh.FIGURES_STR)

WINDOW_SIZES = np.arange(21)
COLORS = {
    "recall": "#2166ac", "far": "#8c510a", "d'": "#762a83",
    "c": "#b35806", "precision": "#01665e", "f1": "#c51b7d",  # comment out if not needed
}

#####################


def load_or_calc_saccade_onset():
    os.makedirs(_FIGURES_DIR, exist_ok=True)
    event_name = "saccade_onset"
    filename_format = "%d_" + event_name
    measures, subject_figures = {}, {}
    for idx in tqdm(range(101, 111), desc="Subjects"):
        # load subject stats
        stats_file_path = os.path.join(_OUTPUT_DIR, filename_format % idx + ".pkl")
        try:
            with open(stats_file_path, "rb") as f:
                measures[idx] = pkl.load(f)
        except FileNotFoundError:
            s = Subject.load_or_make(idx, tavh.OUTPUT_DIR)
            measures[idx] = saccade_event_detection_measures(s, event_name, WINDOW_SIZES)
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


def saccade_event_detection_measures(
        s: Subject, event_name: str, window_sizes: np.ndarray, enforce_trials: bool = True
) -> pd.DataFrame:
    """
    Calculates Signal Detection Theory measures for Saccade Event (onset/offset) detection using Eye-Tracking and REOG
    data. Returns a DataFrame with the calculated measures for each of the specified window sizes.
    Calculated measures include:
        - P: number of positive events in the Ground-Truth
        - N: number of negative events in the Ground-Truth
        - PP: number of positive events in the Predictions
        - TP: number of correctly detected positive events
        - recall: Hit Rate (Sensitivity, Recall, True Positive Rate)
        - precision: Positive Predictive Value (PPV)
        - far: False Alarm Rate (False Positive Rate, Type I Error, 1 - Specificity)
        - f1: F1 Score
        - d': d-prime, Sensitivity Index
        - beta: Response Bias
        - c: Decision Criterion
    """
    # match ET detected events with REOG detected events
    et_event_idxs = s.get_eye_tracking_event_indices(event_name, False)
    et_event_channel = s.create_boolean_event_channel(et_event_idxs, enforce_trials)
    reog_event_idxs = s.calculate_reog_saccade_event_indices(
        event_name=event_name, filter_name='srp', snr=3.5, enforce_trials=False
    )
    reog_event_channel = s.create_boolean_event_channel(reog_event_idxs, enforce_trials)
    P = et_event_channel.sum()      # number of GT positive events
    PP = reog_event_channel.sum()   # number of Predicted positive events
    all_matched_idxs = tavh.match_boolean_events(et_event_channel, reog_event_channel)
    # calculate measures for each window size
    num_trial_idxs = sum(s.get_is_trial_channel())
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
        - recall: Hit Rate (Sensitivity, Recall, True Positive Rate)
        - precision: Positive Predictive Value (PPV)
        - far: False Alarm Rate (False Positive Rate, Type I Error, 1 - Specificity)
        - f1: F1 Score
        - d': d-prime, Sensitivity Index
        - beta: Response Bias
        - c: Decision Criterion
    """
    assert P >= 0 and N >= 0 and PP >= 0 and TP >= 0, "All values must be non-negative"
    assert TP <= P and TP <= PP, "TP must be less than or equal to P and PP"
    hit_rate = TP / P if P > 0 else np.nan           # sensitivity, recall, true positive rate (TPR)
    fa_rate = (PP - TP) / N if N > 0 else np.nan     # false alarm rate, false positive rate (FPR), type I error, 1 - specificity
    precision = TP / PP if PP > 0 else np.nan        # positive predictive value (PPV)
    f1_score = 2 * precision * hit_rate / (precision + hit_rate) if precision + hit_rate > 0 else 0
    d_prime, beta, criterion = _calculate_sdt_measures(P, N, PP, TP, "loglinear")
    return pd.Series({
        'P': P, 'N': N, 'PP': PP, 'TP': TP, "precision": precision, "recall": hit_rate, 'far': fa_rate, "f1": f1_score,
        'd\'': d_prime, 'beta': beta, 'c': criterion,
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
                name=column,
                legendgroup=column,
                showlegend=True,
                mode='lines+markers',
                marker=dict(color=color),
            )
        )
        for stat in stats:
            # add thin lines for individual subjects
            mean_fig.add_trace(
                row=row, col=1,
                trace=go.Scatter(
                    x=window_sizes,
                    y=stat[column],
                    name=column,
                    legendgroup=column,
                    showlegend=False,
                    mode='lines',
                    line=dict(color=color, width=0.2),
                )
            )
    mean_fig.update_layout(
        title="Mean Performance",
        xaxis_title="Window Size (ms)",
        showlegend=True
    )
    return mean_fig


def __calc_rate(true_count: int, detected_count: int) -> float:
    """
    Calculates the Hit Rate / False Alarm Rate while adjusting for floor/ceiling effects by replacing 0 and 1 rates with
    0.25/true_count and 1-0.25/true_count, respectively, as suggested by Macmillan & Kaplan (1985).

    See implementation details at https://lindeloev.net/calculating-d-in-python-and-php/
    See other forms of correction at https://stats.stackexchange.com/a/134802/288290
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
    Calculates d-prime, beta and criterion from Hit Rate and False Alarm Rate, while adjusting for floor/ceiling effects.
    See https://lindeloev.net/calculating-d-in-python-and-php/ for more details.
    """
    assert 0 <= hr <= 1, "Hit Rate must be between 0 and 1"
    assert 0 <= far <= 1, "False Alarm Rate must be between 0 and 1"
    Z = norm.ppf
    d_prime = Z(hr) - Z(far)
    beta = np.exp((Z(far)**2 - Z(hr)**2) / 2)
    criterion = -0.5 * (Z(hr) + Z(far))
    return d_prime, beta, criterion


def _calculate_sdt_measures(
        p: int, n: int, pp: int, tp: int, correction: Optional[str] = "loglinear"
) -> (float, float, float):
    """
    Calculates Signal Detection Theory measures: d-prime, beta and criterion.
    Optionally, adjusts for floor/ceiling effects using the specified correction method.
    See information on correction methods at https://stats.stackexchange.com/a/134802/288290.
    See implementation details at https://lindeloev.net/calculating-d-in-python-and-php/.
    :return: d-prime, beta, criterion
    """
    assert 0 <= p and 0 <= n, "Positive and Negative counts must be non-negative"
    assert 0 <= pp <= p + n, "Predicted Positive count must be between 0 and Total count"
    assert 0 <= tp <= p, "True Positive count must be between 0 and Positive count"
    assert 0 <= tp <= pp, "True Positive count must be between 0 and Predicted Positive count"
    Z = norm.ppf
    hr, far = __calculate_rates_for_sdt(p, n, pp, tp, correction)
    d_prime = Z(hr) - Z(far)
    beta = np.exp((Z(far) ** 2 - Z(hr) ** 2) / 2)
    criterion = -0.5 * (Z(hr) + Z(far))
    return d_prime, beta, criterion


def __calculate_rates_for_sdt(p, n, pp, tp, correction: Optional[str] = None) -> (float, float):
    if correction is None or not correction:
        # correction not specified, return as is
        hr = tp / p if p > 0 else np.nan
        far = (pp - tp) / n if n > 0 else np.nan
        return hr, far
    if correction in {"mk", "m&k", "macmillan-kaplan", "macmillan"}:
        # Macmillan & Kaplan (1985) correction
        hr = __macmillan_kaplan_correction(p, tp)
        far = __macmillan_kaplan_correction(n, pp - tp)
        return hr, far
    if correction in {"ll", "loglinear", "log-linear", "hautus"}:
        # Hautus (1995) correction
        hr, far = __loglinear_correction(p, n, pp, tp)
        return hr, far
    raise ValueError(f"Invalid correction: {correction}")


def __macmillan_kaplan_correction(full_count: int, detected_count: int) -> float:
    """
    Calculates the Hit Rate / False Alarm Rate while adjusting for floor/ceiling effects by replacing 0 and 1 rates with
    0.5/true_count and 1-0.5/true_count, respectively, as suggested by Macmillan & Kaplan (1985).
    See more details at https://stats.stackexchange.com/a/134802/288290.
    Implementation from https://lindeloev.net/calculating-d-in-python-and-php/.
    """
    rate = detected_count / full_count if full_count > 0 else np.nan
    if rate == 0:
        rate = 0.5 / full_count
    if rate == 1:
        rate = 1 - 0.5 / full_count
    return rate


def __loglinear_correction(p, n, pp, tp) -> (float, float):
    """
    Calculates the Hit Rate & False Alarm Rate while adjusting for floor/ceiling effects by adding the proportion of
    positive and negative events to the counts, as suggested by Hautus (1995).
    See https://stats.stackexchange.com/a/134802/288290 for more details.
    """
    fp = pp - tp
    hr = tp / p if p > 0 else np.nan
    far = fp / n if n > 0 else np.nan
    if hr != 0 and hr != 1 and far != 0 and far != 1:
        # no correction needed
        return hr, far
    prevalence = p / (p + n)
    tp, fp = tp + prevalence, fp + 1 - prevalence
    p, n = p + 2 * prevalence, n + 2 * (1 - prevalence)
    hr = tp / p if p > 0 else np.nan
    far = fp / n if n > 0 else np.nan
    return hr, far
