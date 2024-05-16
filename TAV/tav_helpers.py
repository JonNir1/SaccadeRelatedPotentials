import os

import numpy as np
import plotly.graph_objects as go
import scipy.signal as signal
import pywt

OUTPUT_DIR = os.path.join("results", "tav")
FIGURES_STR = "Figures"
SAMPLING_FREQUENCY = 1024  # eeg sampling frequency
SRP_FILTER = np.array([
    0.000e+00, -0.000e+00, -1.000e-04, -2.000e-04, -2.000e-04, -1.000e-04, 1.000e-04, 3.000e-04, 7.000e-04, 1.500e-03,
    2.800e-03, 5.000e-03, 8.000e-03, 1.140e-02, 1.510e-02, 1.880e-02, 2.170e-02, 2.410e-02, 2.670e-02, 2.720e-02,
    2.710e-02, 2.870e-02, 3.290e-02, 3.910e-02, 4.620e-02, 5.440e-02, 6.050e-02, 6.020e-02, 4.470e-02, 3.000e-03,
    -6.720e-02, -1.615e-01, -2.631e-01, -3.490e-01, -3.965e-01, -3.834e-01, -3.045e-01, -1.706e-01, -1.090e-02,
    1.349e-01, 2.355e-01, 2.789e-01, 2.707e-01, 2.271e-01, 1.683e-01, 1.100e-01, 6.310e-02, 3.190e-02, 1.740e-02,
    1.420e-02, 1.930e-02, 2.740e-02, 3.120e-02, 3.030e-02, 2.570e-02, 1.830e-02, 8.800e-03, -7.000e-04, -8.600e-03,
    -1.520e-02, -1.980e-02, -2.210e-02, -2.290e-02, -2.300e-02, -2.190e-02, -1.990e-02, -1.790e-02, -1.570e-02,
    -1.290e-02, -1.010e-02, -7.000e-03, -4.200e-03, -2.000e-03, -3.000e-04, 9.000e-04, 1.300e-03, 1.300e-03, 1.100e-03,
    8.000e-04, 5.000e-04, 2.000e-04, 1.000e-04, 0.000e+00, 0.000e+00
])


def create_filter(name: str) -> (np.ndarray, np.ndarray):
    name = name.lower()
    if name == 'srp':
        return SRP_FILTER, np.ones_like(SRP_FILTER)
    if name == 'butter':
        # TODO
        b, a = signal.butter(6, Wn=np.array([30, 100]), fs=SAMPLING_FREQUENCY, btype='bandpass')
        # return b, a
        raise NotImplementedError
    if name == 'wavelet':
        # TODO
        wavelet = pywt.ContinuousWavelet("gaus1", dtype=float)
        phi, psi, _x = wavelet.wavefun(level=3)
        return phi, psi
    raise ValueError(f"Filter {name} not recognized")


def apply_filter(data: np.ndarray, filter_name: str) -> np.ndarray:
    filter_name = filter_name.lower()
    if filter_name == 'butter':
        # see https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
        b, a = create_filter(filter_name)
        return signal.lfilter(b, a, data)
    if filter_name == 'wavelet':
        # phi, psi = create_filter(filter_name)
        # return signal.convolve(data, psi, mode='same')
        # see https://scicoding.com/introduction-to-wavelet-transform-using-python/
        raise NotImplementedError
    if filter_name == 'srp':
        # copying implementation from Tav's code (see `filterSRP` function in Tav's code)
        srp_filter, _ = create_filter(filter_name)
        n, SPOnset = len(srp_filter), 28
        reog_convolved = np.convolve(data, srp_filter[::-1])
        reog_convolved = reog_convolved[n - SPOnset: 1 - SPOnset]
        return reog_convolved
    raise ValueError(f"Filter {filter_name} not recognized")


def create_boolean_array(s: int, true_indices: np.ndarray) -> np.ndarray:
    """ Creates a boolean array of length `s` with True values at the indices specified in `true_indices`. """
    if true_indices.ndim != 1:
        raise ValueError("true_indices must be one-dimensional")
    if not (0 <= min(true_indices) and max(true_indices) < s):
        raise ValueError(f"true_indices must be within the range [0, {s})")
    bool_array = np.zeros(s, dtype=bool)
    bool_array[true_indices] = True
    return bool_array


def match_events(is_gt: np.ndarray, is_pred: np.ndarray) -> np.ndarray:
    """
    Matches between Ground-Truth and Predicted events, such that there is minimal difference between the indices of
    the matched events. Returns a 2D array where each row is a matching pair of indices, where the first column is the
    GT index and the second column is the Predicted index.
    See also here: https://stackoverflow.com/q/78484847/8543025

    :param is_gt: boolean array indicating the occurrence of Ground-Truth events
    :param is_pred: boolean array indicating the occurrence of Predicted events
    :return: m×2 array of matched indices (0 <= m <= min(sum(is_gt), sum(is_pred))
    """
    gt_idxs = np.where(is_gt)[0]
    pred_idxs = np.where(is_pred)[0]
    diffs = abs(gt_idxs[:, None] - pred_idxs[None, :])
    rowwise_argmin = np.stack([diffs.argmin(0), np.arange(diffs.shape[1])]).T
    colwise_argmin = np.stack([np.arange(diffs.shape[0]), diffs.argmin(1)]).T
    is_matching = (rowwise_argmin[:, None] == colwise_argmin).all(-1).any(1)
    idxs = rowwise_argmin[is_matching]
    matching_indices = np.stack([gt_idxs[idxs[:, 0]], pred_idxs[idxs[:, 1]]]).T
    return matching_indices


def get_output_subdir(analysis_file: str) -> str:
    subdir_name = analysis_file.replace(".py", "").replace("_", " ").title()
    subdir_path = os.path.join(OUTPUT_DIR, subdir_name)
    if not os.path.exists(subdir_path):
        os.makedirs(subdir_path, exist_ok=True)
    return subdir_path



def save_figure(fig: go.Figure, output_dir: str, filename: str):
    fig.write_html(os.path.join(output_dir, f"{filename}.html"))
    fig.write_json(os.path.join(output_dir, f"{filename}.json"))
