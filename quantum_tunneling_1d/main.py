from __future__ import annotations

import argparse
import numpy as np

from tunneling import (
    SimParams,
    square_barrier,
    gaussian_wavepacket,
    CrankNicolson1D,
    transmission_reflection,
    theoretical_tunneling_approx,
    plot_final_state,
    animate_density,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="1D quantum tunneling simulation")
    parser.add_argument("--animate", action="store_true", help="show animation")
    parser.add_argument("--V0", type=float, default=1.5, help="barrier height")
    parser.add_argument("--k0", type=float, default=1.2, help="initial wave number")
    parser.add_argument("--width", type=float, default=10.0, help="barrier width")
    parser.add_argument("--steps", type=int, default=5000, help="number of time steps")
    args = parser.parse_args()

    p = SimParams()
    p.barrier_height = args.V0
    p.k0 = args.k0
    p.barrier_left = -args.width / 2.0
    p.barrier_right = +args.width / 2.0
    p.n_steps = args.steps

    x = np.linspace(p.x_min, p.x_max, p.nx)
    v = square_barrier(x, p.barrier_left, p.barrier_right, p.barrier_height)
    psi0 = gaussian_wavepacket(x, p.x0, p.sigma, p.k0)

    solver = CrankNicolson1D(x, v, p.dt, mass=p.mass, hbar=p.hbar)
    psi_final, snapshots, times = solver.run(psi0, p.n_steps, snapshot_every=p.snapshot_every)

    t_num, r_num = transmission_reflection(x, psi_final, p.barrier_left, p.barrier_right)

    E0 = (p.hbar**2 * p.k0**2) / (2.0 * p.mass)
    L = p.barrier_right - p.barrier_left
    t_approx = theoretical_tunneling_approx(
        barrier_height=p.barrier_height,
        energy=E0,
        barrier_width=L,
        mass=p.mass,
        hbar=p.hbar,
    )

    print("\n=== Quantum Tunneling Simulation ===")
    print(f"Barrier height V0     = {p.barrier_height:.6f}")
    print(f"Barrier width L       = {L:.6f}")
    print(f"Central energy E0     = {E0:.6f}")
    print(f"Numerical T           = {t_num:.6f}")
    print(f"Numerical R           = {r_num:.6f}")
    print(f"T + R                 = {t_num + r_num:.6f}")

    if np.isnan(t_approx):
        print("Approx T (E0 < V0)    = not applicable because E0 >= V0")
    else:
        print(f"Approx T ~ exp(-2κL)  = {t_approx:.6e}")

    if args.animate:
        animate_density(x, v, snapshots, times)
    else:
        plot_final_state(x, v, psi_final)


if __name__ == "__main__":
    main()