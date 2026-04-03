from __future__ import annotations

import json
from pathlib import Path
import numpy as np


def save_samples_json(
    output_path: str | Path,
    n: int,
    l: int,
    m: int,
    data: dict[str, np.ndarray],
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    points = []
    size = len(data["x"])
    for i in range(size):
        points.append(
            {
                "x": float(data["x"][i]),
                "y": float(data["y"][i]),
                "z": float(data["z"][i]),
                "psi_re": float(data["psi_re"][i]),
                "psi_im": float(data["psi_im"][i]),
                "density": float(data["density"][i]),
                "phase": float(data["phase"][i]),
            }
        )

    payload = {
        "n": n,
        "l": l,
        "m": m,
        "points": points,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)