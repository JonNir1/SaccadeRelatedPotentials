from typing import Sequence

import numpy as np


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
