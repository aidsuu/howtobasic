from __future__ import annotations

import math
import numpy as np
import scipy.special as sp

from .constants import A0


def _sph_harm_compat(m: int, l: int, phi: np.ndarray, theta: np.ndarray) -> np.ndarray:
    if hasattr(sp, "sph_harm_y"):
        return sp.sph_harm_y(l, m, theta, phi)
    return sp.sph_harm(m, l, phi, theta)


def radial_wavefunction(n: int, l: int, r: np.ndarray | float, a0: float = A0) -> np.ndarray:
    r = np.asarray(r, dtype=float)

    rho = 2.0 * r / (n * a0)
    norm = math.sqrt(
        (2.0 / (n * a0)) ** 3
        * math.factorial(n - l - 1)
        / (2 * n * math.factorial(n + l))
    )
    lag = sp.assoc_laguerre(rho, n - l - 1, 2 * l + 1)
    return norm * np.exp(-rho / 2.0) * rho**l * lag


def spherical_harmonic(l: int, m: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    return _sph_harm_compat(m, l, phi, theta)


def psi_nlm(
    n: int,
    l: int,
    m: int,
    r: np.ndarray,
    theta: np.ndarray,
    phi: np.ndarray,
    a0: float = A0,
) -> np.ndarray:
    return radial_wavefunction(n, l, r, a0=a0) * spherical_harmonic(l, m, theta, phi)


def psi_real_orbital(
    n: int,
    l: int,
    m: int,
    r: np.ndarray,
    theta: np.ndarray,
    phi: np.ndarray,
    kind: str = "c",
    a0: float = A0,
) -> np.ndarray:
    """
    Orbital real dari kombinasi Y_l^{+m} dan Y_l^{-m}

    kind = "c" -> cosine-like
    kind = "s" -> sine-like
    """
    if m == 0:
        return psi_nlm(n, l, 0, r, theta, phi, a0=a0).real

    psi_p = psi_nlm(n, l, +m, r, theta, phi, a0=a0)
    psi_n = psi_nlm(n, l, -m, r, theta, phi, a0=a0)

    if kind == "c":
        return ((psi_p + ((-1) ** m) * psi_n) / np.sqrt(2.0)).real
    elif kind == "s":
        return ((psi_p - ((-1) ** m) * psi_n) / (1j * np.sqrt(2.0))).real
    else:
        raise ValueError("kind harus 'c' atau 's'")


def spherical_to_cartesian(
    r: np.ndarray,
    theta: np.ndarray,
    phi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z