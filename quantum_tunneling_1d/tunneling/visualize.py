from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def plot_final_state(
    x: np.ndarray,
    v: np.ndarray,
    psi: np.ndarray,
    title: str = "Quantum tunneling",
) -> None:
    density = np.abs(psi) ** 2

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(x, density, label=r"$|\psi|^2$", linewidth=2)
    ax1.set_xlabel("x")
    ax1.set_ylabel(r"$|\psi|^2$")
    ax1.set_title(title)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(x, v, "--", label="V(x)", alpha=0.8)
    ax2.set_ylabel("V(x)")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.tight_layout()
    plt.show()


def animate_density(
    x: np.ndarray,
    v: np.ndarray,
    snapshots: list[np.ndarray],
    times: np.ndarray,
    interval: int = 40,
) -> None:
    densities = [np.abs(psi) ** 2 for psi in snapshots]
    ymax = max(np.max(d) for d in densities) * 1.1

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    line_density, = ax1.plot([], [], lw=2, label=r"$|\psi|^2$")
    line_pot, = ax2.plot(x, v, "--", alpha=0.8, label="V(x)")

    ax1.set_xlim(x[0], x[-1])
    ax1.set_ylim(0, ymax)
    ax2.set_ylim(np.min(v) - 0.1, np.max(v) * 1.2 + 0.1)

    ax1.set_xlabel("x")
    ax1.set_ylabel(r"$|\psi|^2$")
    ax2.set_ylabel("V(x)")
    ax1.grid(True, alpha=0.3)

    title = ax1.set_title("")

    def init():
        line_density.set_data([], [])
        title.set_text("")
        return line_density, line_pot, title

    def update(frame):
        line_density.set_data(x, densities[frame])
        title.set_text(f"Quantum tunneling, t = {times[frame]:.3f}")
        return line_density, line_pot, title

    FuncAnimation(fig, update, frames=len(densities), init_func=init, interval=interval, blit=False)
    plt.tight_layout()
    plt.show()