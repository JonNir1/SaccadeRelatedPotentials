import numpy as np


def calculate_sampling_rate(milliseconds: np.ndarray) -> float:
    """
    Calculates the sampling rate of the given timestamps in Hz.
    :param milliseconds: timestamps in milliseconds (floating-point, not integer)
    """
    if len(milliseconds) < 2:
        raise ValueError("timestamps must be of length at least 2")
    ms_per_sec = 1000
    sr = ms_per_sec / np.median(np.diff(milliseconds))
    if not np.isfinite(sr):
        raise RuntimeError("Error calculating sampling rate")
    return float(sr)
