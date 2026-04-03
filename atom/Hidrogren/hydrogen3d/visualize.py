from __future__ import annotations

from pathlib import Path
import numpy as np
import plotly.graph_objects as go


def make_3d_scatter_figure(
    data: dict[str, np.ndarray],
    title: str = "Hydrogen orbital",
    max_points: int = 30000,
    density_quantile: float = 0.92,
) -> go.Figure:
    x = data["x"]
    y = data["y"]
    z = data["z"]
    density = data["density"]
    psi_re = data["psi_re"]

    cutoff = np.quantile(density, density_quantile)
    mask = density >= cutoff

    x = x[mask]
    y = y[mask]
    z = z[mask]
    density = density[mask]
    psi_re = psi_re[mask]

    if len(x) > max_points:
        idx = np.linspace(0, len(x) - 1, max_points, dtype=int)
        x = x[idx]
        y = y[idx]
        z = z[idx]
        density = density[idx]
        psi_re = psi_re[idx]

    sign_color = np.where(psi_re >= 0.0, 1.0, -1.0)
    brightness = np.log10(density + 1e-20)

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="markers",
                marker=dict(
                    size=2.1,
                    color=brightness,
                    colorscale="Inferno",
                    opacity=0.65,
                    colorbar=dict(title="log10(|ψ|²)"),
                ),
                customdata=np.stack([sign_color], axis=-1),
            )
        ]
    )

    fig.update_layout(
        title=title,
        paper_bgcolor="black",
        plot_bgcolor="black",
        font=dict(color="white"),
        scene=dict(
            xaxis=dict(
                title="x",
                backgroundcolor="black",
                gridcolor="rgba(255,255,255,0.2)",
                zerolinecolor="rgba(255,255,255,0.3)",
                color="white",
            ),
            yaxis=dict(
                title="y",
                backgroundcolor="black",
                gridcolor="rgba(255,255,255,0.2)",
                zerolinecolor="rgba(255,255,255,0.3)",
                color="white",
            ),
            zaxis=dict(
                title="z",
                backgroundcolor="black",
                gridcolor="rgba(255,255,255,0.2)",
                zerolinecolor="rgba(255,255,255,0.3)",
                color="white",
            ),
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=50, b=0),
    )

    return fig


def save_html(fig: go.Figure, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path))