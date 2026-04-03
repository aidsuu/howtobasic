from __future__ import annotations

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import splu


class CrankNicolson1D:
    """
    Solve:
        i hbar dpsi/dt = [-(hbar^2/2m)d2/dx2 + V(x)] psi
    with Dirichlet boundary psi=0 at edges.
    """

    def __init__(self, x: np.ndarray, v: np.ndarray, dt: float, mass: float = 1.0, hbar: float = 1.0):
        self.x = x
        self.v = v
        self.dt = dt
        self.mass = mass
        self.hbar = hbar

        self.dx = x[1] - x[0]
        self.n = len(x)

        self._build_matrices()

    def _build_matrices(self) -> None:
        n = self.n
        dx = self.dx
        dt = self.dt
        m = self.mass
        hbar = self.hbar

        lap_main = -2.0 * np.ones(n)
        lap_off = 1.0 * np.ones(n - 1)

        # second derivative matrix
        d2 = diags([lap_off, lap_main, lap_off], offsets=[-1, 0, 1], shape=(n, n), dtype=complex) / dx**2

        h_kin = -(hbar**2) / (2.0 * m) * d2
        h_pot = diags(self.v.astype(complex), offsets=0)
        h = h_kin + h_pot

        ident = diags([np.ones(n)], [0], dtype=complex)

        a = ident + 1j * dt / (2.0 * hbar) * h
        b = ident - 1j * dt / (2.0 * hbar) * h

        # hard Dirichlet edges
        a = a.tolil()
        b = b.tolil()

        for i in [0, n - 1]:
            a[i, :] = 0.0
            a[i, i] = 1.0
            b[i, :] = 0.0
            b[i, i] = 1.0

        self.a_lu = splu(a.tocsc())
        self.b = b.tocsc()

    def step(self, psi: np.ndarray) -> np.ndarray:
        rhs = self.b @ psi
        rhs[0] = 0.0
        rhs[-1] = 0.0
        return self.a_lu.solve(rhs)

    def run(
        self,
        psi0: np.ndarray,
        n_steps: int,
        snapshot_every: int = 20,
    ) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
        psi = psi0.copy()
        snapshots = [psi.copy()]
        times = [0.0]

        for step in range(1, n_steps + 1):
            psi = self.step(psi)

            # renormalize slightly to control numerical drift
            norm = np.sqrt(np.trapezoid(np.abs(psi) ** 2, self.x))
            if norm > 0:
                psi /= norm

            if step % snapshot_every == 0:
                snapshots.append(psi.copy())
                times.append(step * self.dt)

        return psi, snapshots, np.array(times)


def transmission_reflection(
    x: np.ndarray,
    psi: np.ndarray,
    barrier_left: float,
    barrier_right: float,
) -> tuple[float, float]:
    density = np.abs(psi) ** 2

    left_mask = x < barrier_left
    right_mask = x > barrier_right

    r = np.trapezoid(density[left_mask], x[left_mask])
    t = np.trapezoid(density[right_mask], x[right_mask])

    return float(t), float(r)


def theoretical_tunneling_approx(
    barrier_height: float,
    energy: float,
    barrier_width: float,
    mass: float = 1.0,
    hbar: float = 1.0,
) -> float:
    if energy >= barrier_height:
        return float("nan")

    kappa = np.sqrt(2.0 * mass * (barrier_height - energy)) / hbar
    return float(np.exp(-2.0 * kappa * barrier_width))