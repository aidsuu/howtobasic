from __future__ import annotations

import numpy as np

from .constants import A0
from .wavefunction import (
    radial_wavefunction,
    spherical_harmonic,
    psi_real_orbital,
    spherical_to_cartesian,
)


def _build_cdf(pdf: np.ndarray) -> np.ndarray:
    cdf = np.cumsum(pdf)
    cdf /= cdf[-1]
    return cdf


def _sample_from_cdf(grid: np.ndarray, cdf: np.ndarray, size: int, rng: np.random.Generator) -> np.ndarray:
    u = rng.random(size)
    return np.interp(u, cdf, grid)


def sample_radial(
    n: int,
    l: int,
    size: int,
    a0: float = A0,
    r_max: float | None = None,
    n_grid: int = 4000,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Sampling radial dari distribusi:
    P(r) dr ∝ r^2 |R_{nl}(r)|^2 dr
    """
    if rng is None:
        rng = np.random.default_rng()

    if r_max is None:
        r_max = 12.0 * n * n * a0

    r_grid = np.linspace(0.0, r_max, n_grid)
    R = radial_wavefunction(n, l, r_grid, a0=a0)
    pdf = (r_grid**2) * np.abs(R) ** 2
    pdf += 1e-300
    cdf = _build_cdf(pdf)
    return _sample_from_cdf(r_grid, cdf, size, rng)


def sample_theta_phi(
    l: int,
    m: int,
    size: int,
    n_theta: int = 3000,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sampling angular dari distribusi:
    P(theta, phi) dtheta dphi ∝ |Y_l^m(theta,phi)|^2 sin(theta) dtheta dphi

    Karena |Y_l^m|^2 tidak bergantung pada phi untuk satu state (n,l,m),
    maka phi uniform, sedangkan theta dari:
    P(theta) dtheta ∝ |Y_l^m(theta,phi)|^2 sin(theta) dtheta
    """
    if rng is None:
        rng = np.random.default_rng()

    theta_grid = np.linspace(0.0, np.pi, n_theta)

    # phi bebas karena |exp(i m phi)|^2 = 1
    phi0 = np.zeros_like(theta_grid)

    Y = spherical_harmonic(l, m, theta_grid, phi0)
    pdf_theta = np.abs(Y) ** 2 * np.sin(theta_grid)
    pdf_theta += 1e-300
    cdf_theta = _build_cdf(pdf_theta)

    theta = _sample_from_cdf(theta_grid, cdf_theta, size, rng)
    phi = rng.uniform(0.0, 2.0 * np.pi, size)

    return theta, phi


def sample_orbital_points(
    n: int,
    l: int,
    m: int,
    n_samples: int = 50_000,
    a0: float = A0,
    seed: int | None = None,
    real_orbital: bool = True,
    real_kind: str = "c",
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)

    r = sample_radial(n=n, l=l, size=n_samples, a0=a0, rng=rng)
    theta, phi = sample_theta_phi(l=l, m=m, size=n_samples, rng=rng)

    x, y, z = spherical_to_cartesian(r, theta, phi)

    if real_orbital:
        psi = psi_real_orbital(n, l, m, r, theta, phi, kind=real_kind, a0=a0)
        psi_re = np.real(psi)
        psi_im = np.zeros_like(psi_re)
        density = np.abs(psi) ** 2
        phase = np.where(psi_re >= 0.0, 0.0, np.pi)
    else:
        psi = psi_nlm(n, l, m, r, theta, phi, a0=a0)
        psi_re = np.real(psi)
        psi_im = np.imag(psi)
        density = np.abs(psi) ** 2
        phase = np.angle(psi)

    return {
        "r": r,
        "theta": theta,
        "phi": phi,
        "x": x,
        "y": y,
        "z": z,
        "psi_re": psi_re,
        "psi_im": psi_im,
        "density": density,
        "phase": phase,
    }