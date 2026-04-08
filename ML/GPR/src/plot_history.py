#!/usr/bin/env python3
"""Plot training history from outputs/history.csv."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MPLCONFIGDIR = PROJECT_ROOT / ".mplconfig"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib.pyplot as plt


def main() -> int:
    history_path = (PROJECT_ROOT / "outputs/history.csv").resolve()
    output_path = (PROJECT_ROOT / "outputs/history_plot.png").resolve()

    if not history_path.exists():
        print(f"History CSV not found: {history_path}", file=sys.stderr)
        return 1

    try:
        history = pd.read_csv(history_path)
    except Exception as exc:
        print(f"Failed to read history CSV: {exc}", file=sys.stderr)
        return 1

    required_columns = {"epoch", "train_loss", "val_loss", "val_mae", "val_rmse"}
    missing = required_columns.difference(history.columns)
    if missing:
        print(f"History CSV is missing columns: {sorted(missing)}", file=sys.stderr)
        return 1

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)

    axes[0].plot(history["epoch"], history["train_loss"], marker="o", label="train_loss")
    axes[0].plot(history["epoch"], history["val_loss"], marker="o", label="val_loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history["epoch"], history["val_mae"], marker="o", label="val_mae")
    axes[1].plot(history["epoch"], history["val_rmse"], marker="o", label="val_rmse")
    axes[1].set_title("Validation Metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Error")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"history_csv={history_path}")
    print(f"history_plot={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
