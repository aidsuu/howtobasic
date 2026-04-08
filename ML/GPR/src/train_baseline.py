#!/usr/bin/env python3
"""Minimal end-to-end baseline training script for MERL-GPR."""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Subset

from dataset_merl import MerlDataset, VALID_INPUT_MODES
from model_baseline import SmallEncoderDecoder


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MPLCONFIGDIR = PROJECT_ROOT / ".mplconfig"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a minimal CNN baseline for MERL-GPR sanity checks."
    )
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=8, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count.")
    parser.add_argument(
        "--input-mode",
        type=str,
        default="magnitude",
        choices=sorted(VALID_INPUT_MODES),
        help="Dataset input representation mode.",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optional cap on the number of train samples for faster sanity checks.",
    )
    parser.add_argument(
        "--max-val-samples",
        type=int,
        default=None,
        help="Optional cap on the number of validation samples for faster sanity checks.",
    )
    parser.add_argument("--seed", type=int, default=20260407, help="Random seed.")
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=None,
        help="Stop if validation loss does not improve beyond min-delta for this many epochs.",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=0.0,
        help="Minimum val_loss improvement required to reset early stopping patience.",
    )
    parser.add_argument(
        "--torch-num-threads",
        type=int,
        default=None,
        help="Optional number of intra-op PyTorch CPU threads.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device selection policy.",
    )
    return parser.parse_args()


def print_header(title: str) -> None:
    print(f"\n{'=' * 80}")
    print(title)
    print("=" * 80)


def set_seed(seed: int, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Device 'cuda' was requested, but torch.cuda.is_available() is False."
            )
        return torch.device("cuda")
    raise ValueError(f"Unsupported device argument: {device_arg}")


def resolve_output_path(relative_path: str) -> Path:
    return (PROJECT_ROOT / relative_path).resolve()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def limit_dataset(dataset: MerlDataset, max_samples: Optional[int]) -> torch.utils.data.Dataset:
    if max_samples is None:
        return dataset
    if max_samples <= 0:
        raise ValueError("max sample limits must be > 0 when provided.")
    count = min(len(dataset), max_samples)
    return Subset(dataset, list(range(count)))


def build_dataloader(
    split: str,
    input_mode: str,
    batch_size: int,
    num_workers: int,
    max_samples: Optional[int],
    device: torch.device,
) -> tuple[torch.utils.data.Dataset, DataLoader]:
    dataset = MerlDataset(split=split, input_mode=input_mode, normalize_input=True)
    dataset_or_subset = limit_dataset(dataset, max_samples)
    loader = DataLoader(
        dataset_or_subset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    return dataset_or_subset, loader


def close_dataset(dataset: torch.utils.data.Dataset) -> None:
    if hasattr(dataset, "close"):
        dataset.close()
        return
    if isinstance(dataset, Subset):
        close_dataset(dataset.dataset)


def infer_input_channels(loader: DataLoader) -> tuple[int, tuple[int, ...], tuple[int, ...]]:
    try:
        first_batch = next(iter(loader))
    except StopIteration as exc:
        raise RuntimeError("The DataLoader is empty.") from exc

    input_tensor = first_batch["input"]
    target_tensor = first_batch["target"]
    if input_tensor.ndim != 4:
        raise ValueError(f"Expected input batch to be 4D, got shape {tuple(input_tensor.shape)}")
    if target_tensor.ndim != 4:
        raise ValueError(f"Expected target batch to be 4D, got shape {tuple(target_tensor.shape)}")

    return int(input_tensor.shape[1]), tuple(input_tensor.shape), tuple(target_tensor.shape)


def move_batch_to_device(batch: Dict[str, object], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, List[str]]:
    inputs = batch["input"]
    targets = batch["target"]
    sample_ids = batch["sample_id"]

    if not isinstance(inputs, torch.Tensor) or not isinstance(targets, torch.Tensor):
        raise TypeError("Batch input and target must be torch.Tensor instances.")

    return (
        inputs.to(device, non_blocking=True),
        targets.to(device, non_blocking=True),
        list(sample_ids),
    )


def validate_output_shape(predictions: torch.Tensor, targets: torch.Tensor) -> None:
    if tuple(predictions.shape) != tuple(targets.shape):
        raise RuntimeError(
            "Model output shape does not match target shape: "
            f"output={tuple(predictions.shape)}, target={tuple(targets.shape)}"
        )


def run_training_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    total_samples = 0

    for batch in loader:
        inputs, targets, _sample_ids = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)

        predictions = model(inputs)
        validate_output_shape(predictions, targets)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()

        batch_size = inputs.shape[0]
        running_loss += float(loss.item()) * batch_size
        total_samples += batch_size

    if total_samples == 0:
        raise RuntimeError("Training loader produced zero samples.")

    return running_loss / total_samples


def run_validation_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float, List[Dict[str, np.ndarray]]]:
    model.eval()
    running_loss = 0.0
    running_abs_error = 0.0
    running_squared_error = 0.0
    total_values = 0
    total_samples = 0
    examples: List[Dict[str, np.ndarray]] = []

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

            for idx in range(batch_size):
                if len(examples) >= 3:
                    break
                examples.append(
                    {
                        "sample_id": sample_ids[idx],
                        "input": inputs[idx].detach().cpu().numpy(),
                        "target": targets[idx].detach().cpu().numpy(),
                        "prediction": predictions[idx].detach().cpu().numpy(),
                    }
                )

    if total_samples == 0 or total_values == 0:
        raise RuntimeError("Validation loader produced zero samples.")

    val_loss = running_loss / total_samples
    mae = running_abs_error / total_values
    rmse = math.sqrt(running_squared_error / total_values)
    return val_loss, mae, rmse, examples


def save_history_csv(history: List[Dict[str, float]], output_csv: Path) -> None:
    ensure_parent(output_csv)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["epoch", "train_loss", "val_loss", "val_mae", "val_rmse"],
        )
        writer.writeheader()
        for row in history:
            writer.writerow(row)


def save_best_model(
    model: nn.Module,
    model_path: Path,
    input_mode: str,
    best_val_loss: float,
    epoch: int,
    seed: int,
) -> None:
    ensure_parent(model_path)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_mode": input_mode,
            "best_val_loss": best_val_loss,
            "epoch": epoch,
            "seed": seed,
        },
        model_path,
    )


def save_validation_examples(examples: List[Dict[str, np.ndarray]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for example in examples[:3]:
        sample_id = example["sample_id"]
        input_map = example["input"][0]
        target_map = example["target"][0]
        prediction_map = example["prediction"][0]

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
        panels = [
            ("Input: channel 0", input_map, "viridis"),
            ("Target: eps", target_map, "magma"),
            ("Prediction", prediction_map, "magma"),
        ]

        for axis, (title, image, cmap) in zip(axes, panels):
            im = axis.imshow(image, cmap=cmap)
            axis.set_title(title)
            axis.set_xticks([])
            axis.set_yticks([])
            fig.colorbar(im, ax=axis, fraction=0.046, pad=0.04)

        output_path = output_dir / f"val_sample_{sample_id}.png"
        fig.savefig(output_path, dpi=150)
        plt.close(fig)


def main() -> int:
    args = parse_args()

    if args.epochs <= 0:
        print("epochs must be > 0", file=sys.stderr)
        return 1
    if args.batch_size <= 0:
        print("batch-size must be > 0", file=sys.stderr)
        return 1
    if args.lr <= 0:
        print("lr must be > 0", file=sys.stderr)
        return 1
    if args.num_workers < 0:
        print("num-workers must be >= 0", file=sys.stderr)
        return 1
    if args.early_stopping_patience is not None and args.early_stopping_patience <= 0:
        print("early-stopping-patience must be > 0 when provided", file=sys.stderr)
        return 1
    if args.min_delta < 0:
        print("min-delta must be >= 0", file=sys.stderr)
        return 1

    try:
        device = resolve_device(args.device)
    except Exception as exc:
        print(f"Failed to resolve device: {exc}", file=sys.stderr)
        return 1

    if args.torch_num_threads is not None:
        if args.torch_num_threads <= 0:
            print("torch-num-threads must be > 0 when provided", file=sys.stderr)
            return 1
        torch.set_num_threads(args.torch_num_threads)

    set_seed(args.seed, device=device)

    best_model_path = resolve_output_path("models/best_model.pt")
    history_path = resolve_output_path("outputs/history.csv")
    val_examples_dir = resolve_output_path("outputs/val_examples")

    print_header("Configuration")
    print(f"device={device}")
    print(f"device_arg={args.device}")
    print(f"epochs={args.epochs}")
    print(f"batch_size={args.batch_size}")
    print(f"lr={args.lr}")
    print(f"num_workers={args.num_workers}")
    print(f"input_mode={args.input_mode}")
    print(f"max_train_samples={args.max_train_samples}")
    print(f"max_val_samples={args.max_val_samples}")
    print(f"seed={args.seed}")
    print(f"early_stopping_patience={args.early_stopping_patience}")
    print(f"min_delta={args.min_delta}")
    print(f"torch_num_threads={torch.get_num_threads()}")
    print(f"best_model_path={best_model_path}")
    print(f"history_path={history_path}")
    print(f"val_examples_dir={val_examples_dir}")

    try:
        train_dataset, train_loader = build_dataloader(
            split="train",
            input_mode=args.input_mode,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_samples=args.max_train_samples,
            device=device,
        )
        val_dataset, val_loader = build_dataloader(
            split="val",
            input_mode=args.input_mode,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_samples=args.max_val_samples,
            device=device,
        )
    except Exception as exc:
        print(f"Failed to build datasets/loaders: {exc}", file=sys.stderr)
        return 1

    try:
        in_channels, first_input_shape, first_target_shape = infer_input_channels(train_loader)
    except Exception as exc:
        print(f"Failed to inspect the first training batch: {exc}", file=sys.stderr)
        return 1

    print_header("First Batch Shapes")
    print(f"train_input_shape={first_input_shape}")
    print(f"train_target_shape={first_target_shape}")

    model = SmallEncoderDecoder(in_channels=in_channels, out_channels=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    print_header("Model Summary")
    print(f"in_channels={in_channels}")
    print(f"train_samples={len(train_dataset)}")
    print(f"val_samples={len(val_dataset)}")

    history: List[Dict[str, float]] = []
    best_val_loss = float("inf")
    best_examples: List[Dict[str, np.ndarray]] = []
    best_epoch = 0
    early_stopping_triggered = False
    epochs_without_improvement = 0

    try:
        for epoch in range(1, args.epochs + 1):
            train_loss = run_training_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
            )
            val_loss, val_mae, val_rmse, val_examples = run_validation_epoch(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
            )

            history_row = {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "val_mae": float(val_mae),
                "val_rmse": float(val_rmse),
            }
            history.append(history_row)

            print(
                f"epoch={epoch:03d} "
                f"train_loss={train_loss:.6f} "
                f"val_loss={val_loss:.6f} "
                f"val_mae={val_mae:.6f} "
                f"val_rmse={val_rmse:.6f}"
            )

            improvement = best_val_loss - val_loss
            if improvement > args.min_delta:
                best_val_loss = val_loss
                best_examples = val_examples
                best_epoch = epoch
                epochs_without_improvement = 0
                save_best_model(
                    model=model,
                    model_path=best_model_path,
                    input_mode=args.input_mode,
                    best_val_loss=best_val_loss,
                    epoch=epoch,
                    seed=args.seed,
                )
                print(f"saved_best_model epoch={epoch} val_loss={val_loss:.6f}")
            else:
                epochs_without_improvement += 1
                print(
                    "no_significant_improvement "
                    f"epoch={epoch} "
                    f"epochs_without_improvement={epochs_without_improvement}"
                )

            if (
                args.early_stopping_patience is not None
                and epochs_without_improvement >= args.early_stopping_patience
            ):
                early_stopping_triggered = True
                print(
                    "early_stopping_triggered "
                    f"epoch={epoch} "
                    f"best_epoch={best_epoch} "
                    f"best_val_loss={best_val_loss:.6f} "
                    f"patience={args.early_stopping_patience} "
                    f"min_delta={args.min_delta}"
                )
                break
    except Exception as exc:
        print(f"Training failed: {exc}", file=sys.stderr)
        close_dataset(train_dataset)
        close_dataset(val_dataset)
        return 1

    save_history_csv(history, history_path)
    save_validation_examples(best_examples, val_examples_dir)

    close_dataset(train_dataset)
    close_dataset(val_dataset)

    print_header("Finished")
    print(f"best_epoch={best_epoch}")
    print(f"best_val_loss={best_val_loss:.6f}")
    print(f"early_stopping_triggered={early_stopping_triggered}")
    print(f"history_saved={history_path}")
    print(f"best_model_saved={best_model_path}")
    print(f"val_examples_saved={val_examples_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
