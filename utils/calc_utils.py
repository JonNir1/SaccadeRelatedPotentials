import numpy as np


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
