import numpy as np


def gaussian_wavepacket(
    x: np.ndarray,
    x0: float,
    sigma: float,
    k0: float,
) -> np.ndarray:
    psi = np.exp(-((x - x0) ** 2) / (4.0 * sigma**2)) * np.exp(1j * k0 * x)
    norm = np.sqrt(np.trapezoid(np.abs(psi) ** 2, x))
    return psi / norm