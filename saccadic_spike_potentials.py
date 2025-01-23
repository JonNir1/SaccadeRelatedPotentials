import numpy as np
import scipy as sp

import utils.array_utils as au


_SP_FILTER = np.array([
    # spike potential filter from Keren, Yuval-Greenberg & Deouell  (2010): https://doi.org/10.1016/j.neuroimage.2009.10.057
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
_SP_ONSET = 28  # onset of the saccadic spike potential (SP) in the filter


def detect_saccadic_spike_potentials(
        left_eog: np.ndarray,
        right_eog: np.ndarray,
        top_eog: np.ndarray,
        bottom_eog: np.ndarray,
        reference: np.ndarray,
        snr: float = 3.5,
        min_distance: int = None,
) -> np.ndarray:
    """
    Calculated the radial EOG signal from the input EOGs and reference signal, applies a spike potential filter to the
    radial EOG signal, and detects the indices of the saccadic spike potentials (SPs) in the filtered signal.
    All EOG and reference signals must be 1D arrays of the same length.

    :param left_eog: left EOG signal
    :param right_eog: right EOG signal
    :param top_eog: top EOG signal
    :param bottom_eog: bottom EOG signal
    :param reference: reference signal
    :param snr: signal-to-noise ratio, must be a positive number
    :param min_distance: minimum distance between peaks, must be a positive integer

    :return: indices of detected SPs
    """
    reog = _calculate_radial_eog(left_eog, right_eog, top_eog, bottom_eog, reference)
    filtered_reog = _apply_sp_filter(reog)
    idxs = _detect_spike_indices(filtered_reog, snr, min_distance)
    return idxs


def _calculate_radial_eog(
        left_eog: np.ndarray,
        right_eog: np.ndarray,
        top_eog: np.ndarray,
        bottom_eog: np.ndarray,
        reference: np.ndarray,
) -> np.ndarray:
    left_eog = au.to_vector(left_eog)
    right_eog = au.to_vector(right_eog)
    top_eog = au.to_vector(top_eog)
    bottom_eog = au.to_vector(bottom_eog)
    reference = au.to_vector(reference)
    if len(left_eog) != len(right_eog) or len(left_eog) != len(top_eog) or len(left_eog) != len(bottom_eog) or len(
            left_eog) != len(reference):
        raise ValueError("All EOG signals must have the same length.")
    mean_eog = np.nanmean(np.vstack([left_eog, right_eog, top_eog, bottom_eog]), axis=0)
    return mean_eog - reference


def _apply_sp_filter(reog: np.ndarray) -> np.ndarray:
    convolved = np.convolve(reog, _SP_FILTER[::-1])
    convolved = convolved[len(_SP_FILTER) - _SP_ONSET : 1 - _SP_ONSET]
    return convolved


def _detect_spike_indices(
        data: np.ndarray, snr: float = 3.5, min_distance: int = None
) -> np.ndarray:
    """
    Detects indices of peaks in the input array.
    See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html

    :param data: input 1D array
    :param snr: signal-to-noise ratio
    :param min_distance: minimum distance between peaks
    :return: indices of detected peaks
    """
    if snr <= 0:
        raise ValueError("Signal-to-Noise Ratio (SNR) must be a positive number.")
    if min_distance is not None and min_distance <= 0:
        raise ValueError("Minimum distance between peaks must be a positive number.")
    data = au.to_vector(data)
    minimum_peak_height = np.mean(data) + snr * np.std(data)
    peak_idxs, _properties = sp.signal.find_peaks(data, height=minimum_peak_height, distance=min_distance)
    return peak_idxs
