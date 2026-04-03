import numpy as np


def square_barrier(x: np.ndarray, left: float, right: float, height: float) -> np.ndarray:
    v = np.zeros_like(x)
    mask = (x >= left) & (x <= right)
    v[mask] = height
    return v