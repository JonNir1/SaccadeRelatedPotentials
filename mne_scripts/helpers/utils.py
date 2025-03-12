from typing import Sequence

import numpy as np

MILLISECONDS_IN_SECOND = 1000


def milliseconds_to_samples(ms: float, sfreq: float) -> int:
    """ Converts milliseconds to samples given the sampling frequency. """
    assert ms >= 0, "milliseconds must be non-negative"
    assert sfreq > 0, "sampling frequency must be positive"
    return int(round(ms * sfreq / MILLISECONDS_IN_SECOND, 0))


def to_vector(seq: Sequence) -> np.ndarray:
    arr = np.array(seq)
    if arr.ndim == 1:
        return arr
    if arr.ndim >= 3:
        raise TypeError(f"Cannot convert an array with {arr.ndim} dimensions to a vector.")
    # reached here if array.ndim == 2
    assert arr.ndim == 2
    rows, cols = arr.shape
    if rows == 1:
        return arr[0]
    if cols == 1:
        return arr[:, 0]
    raise TypeError(f"Cannot convert a 2D matrix with {rows} rows and {cols} columns to a vector.")


def calculate_sampling_rate(milliseconds: np.ndarray, decimals: int = None) -> float:
    """
    Calculates the sampling rate of the given timestamps in Hz.
    :param milliseconds: timestamps in milliseconds (floating-point, not integer)
    :param decimals: number of decimal places to round to
    """
    if len(milliseconds) < 2:
        raise ValueError("timestamps must be of length at least 2")
    if decimals is not None and not isinstance(decimals, int):
        raise TypeError("decimals must be an integer")
    if decimals is not None and decimals < 0:
        raise ValueError("decimals must be non-negative")
    ms_per_sec = 1000
    sr = ms_per_sec / np.median(np.diff(milliseconds))
    if not np.isfinite(sr):
        raise RuntimeError("Error calculating sampling rate")
    if decimals is None:
        return float(sr)
    return round(sr, decimals)
