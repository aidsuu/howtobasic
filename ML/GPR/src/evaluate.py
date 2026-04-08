#!/usr/bin/env python3
"""Evaluate the baseline MERL-GPR model on the test split."""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset_merl import MerlDataset, VALID_INPUT_MODES
from model_baseline import SmallEncoderDecoder
from train_baseline import (
    PROJECT_ROOT,
    close_dataset,
    move_batch_to_device,
    resolve_device,
    validate_output_shape,
)


MPLCONFIGDIR = PROJECT_ROOT / ".mplconfig"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the MERL-GPR baseline model on the test split."
    )
    parser.add_argument(
        "--input-mode",
        type=str,
        default="magnitude",
        choices=sorted(VALID_INPUT_MODES),
        help="Dataset input representation mode.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device selection policy.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Evaluation batch size.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker count.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/best_model.pt"),
        help="Path to the saved baseline checkpoint.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path("outputs/test_examples"),
        help="Directory where the best test visualizations will be saved.",
    )
    parser.add_argument(
        "--num-save-examples",
        type=int,
        default=10,
        help="Number of best test examples to save.",
    )
    return parser.parse_args()


def print_header(title: str) -> None:
    print(f"\n{'=' * 80}")
    print(title)
    print("=" * 80)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float, list[dict[str, object]]]:
    model.eval()
    running_loss = 0.0
    running_abs_error = 0.0
    running_squared_error = 0.0
    total_values = 0
    total_samples = 0
    sample_summaries: list[dict[str, object]] = []

    with torch.no_grad():
        for batch in loader:
            inputs, targets, sample_ids = move_batch_to_device(batch, device)
            predictions = model(inputs)
            validate_output_shape(predictions, targets)

            loss = criterion(predictions, targets)
            abs_error = torch.abs(predictions - targets)
            squared_error = torch.square(predictions - targets)

            batch_size = inputs.shape[0]
            running_loss += float(loss.item()) * batch_size
            running_abs_error += float(abs_error.sum().item())
            running_squared_error += float(squared_error.sum().item())
            total_values += int(targets.numel())
            total_samples += batch_size

            sample_mse = torch.mean(torch.square(predictions - targets), dim=(1, 2, 3))
            for idx in range(batch_size):
                sample_summaries.append(
                    {
                        "sample_id": sample_ids[idx],
                        "sample_mse": float(sample_mse[idx].item()),
                        "input": inputs[idx].detach().cpu().numpy(),
                        "target": targets[idx].detach().cpu().numpy(),
                        "prediction": predictions[idx].detach().cpu().numpy(),
                    }
                )

    if total_samples == 0 or total_values == 0:
        raise RuntimeError("Evaluation loader produced zero samples.")

    mse = running_loss / total_samples
    mae = running_abs_error / total_values
    rmse = math.sqrt(running_squared_error / total_values)
    return mse, mae, rmse, sample_summaries


def save_test_visualizations(
    sample_summaries: list[dict[str, object]],
    output_dir: Path,
    num_examples: int,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    best_samples = sorted(sample_summaries, key=lambda item: float(item["sample_mse"]))[:num_examples]
    saved_paths: list[Path] = []

    for rank, sample in enumerate(best_samples, start=1):
        sample_id = str(sample["sample_id"])
        input_array = np.asarray(sample["input"])
        target_array = np.asarray(sample["target"])
        prediction_array = np.asarray(sample["prediction"])

        input_map = input_array[0]
        target_map = target_array[0]
        prediction_map = prediction_array[0]
        error_map = np.abs(prediction_map - target_map)

        fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), constrained_layout=True)
        panels = [
            ("Input: channel 0", input_map, "viridis"),
            ("Target: eps", target_map, "magma"),
            ("Prediction", prediction_map, "magma"),
            ("Absolute error", error_map, "inferno"),
        ]

        for axis, (title, image, cmap) in zip(axes, panels):
            im = axis.imshow(image, cmap=cmap)
            axis.set_title(title)
            axis.set_xticks([])
            axis.set_yticks([])
            fig.colorbar(im, ax=axis, fraction=0.046, pad=0.04)

        fig.suptitle(
            f"rank={rank} sample_id={sample_id} sample_mse={float(sample['sample_mse']):.6f}",
            fontsize=11,
        )
        output_path = output_dir / f"test_rank_{rank:02d}_sample_{sample_id}.png"
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        saved_paths.append(output_path)

    return saved_paths


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0:
        print("batch-size must be > 0", file=sys.stderr)
        return 1
    if args.num_workers < 0:
        print("num-workers must be >= 0", file=sys.stderr)
        return 1
    if args.num_save_examples <= 0:
        print("num-save-examples must be > 0", file=sys.stderr)
        return 1

    try:
        device = resolve_device(args.device)
    except Exception as exc:
        print(f"Failed to resolve device: {exc}", file=sys.stderr)
        return 1

    checkpoint_path = (PROJECT_ROOT / args.checkpoint).resolve() if not args.checkpoint.is_absolute() else args.checkpoint.resolve()
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}", file=sys.stderr)
        return 1

    print_header("Evaluation Configuration")
    print(f"device={device}")
    print(f"device_arg={args.device}")
    print(f"input_mode={args.input_mode}")
    print(f"checkpoint={checkpoint_path}")
    print(f"save_dir={(PROJECT_ROOT / args.save_dir).resolve() if not args.save_dir.is_absolute() else args.save_dir.resolve()}")
    print(f"num_save_examples={args.num_save_examples}")

    try:
        dataset = MerlDataset(split="test", input_mode=args.input_mode, normalize_input=True)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )
        first_batch = next(iter(loader))
    except Exception as exc:
        print(f"Failed to build evaluation dataset/loader: {exc}", file=sys.stderr)
        return 1

    input_tensor = first_batch["input"]
    target_tensor = first_batch["target"]
    if input_tensor.ndim != 4 or target_tensor.ndim != 4:
        print(
            f"Unexpected batch shape: input={tuple(input_tensor.shape)}, target={tuple(target_tensor.shape)}",
            file=sys.stderr,
        )
        close_dataset(dataset)
        return 1

    print_header("First Batch Shapes")
    print(f"test_input_shape={tuple(input_tensor.shape)}")
    print(f"test_target_shape={tuple(target_tensor.shape)}")

    in_channels = int(input_tensor.shape[1])
    model = SmallEncoderDecoder(in_channels=in_channels, out_channels=1).to(device)

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    except Exception as exc:
        close_dataset(dataset)
        print(f"Failed to load checkpoint: {exc}", file=sys.stderr)
        return 1

    criterion = nn.MSELoss()

    try:
        mse, mae, rmse, sample_summaries = evaluate(
            model=model,
            loader=loader,
            criterion=criterion,
            device=device,
        )
    except Exception as exc:
        close_dataset(dataset)
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 1

    close_dataset(dataset)

    save_dir = (PROJECT_ROOT / args.save_dir).resolve() if not args.save_dir.is_absolute() else args.save_dir.resolve()
    try:
        saved_paths = save_test_visualizations(
            sample_summaries=sample_summaries,
            output_dir=save_dir,
            num_examples=args.num_save_examples,
        )
    except Exception as exc:
        print(f"Failed to save test visualizations: {exc}", file=sys.stderr)
        return 1

    print_header("Test Metrics")
    print(f"test_mse={mse:.6f}")
    print(f"test_mae={mae:.6f}")
    print(f"test_rmse={rmse:.6f}")
    print(f"saved_test_examples={len(saved_paths)}")
    for path in saved_paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
