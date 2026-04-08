#!/usr/bin/env python3
"""Debug loader for MERL-GPR dataset samples."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset_merl import MerlDataset, VALID_INPUT_MODES


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_MPLCONFIGDIR = _PROJECT_ROOT / ".mplconfig"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a few MERL-GPR samples and save basic debug visualizations."
    )
    parser.add_argument(
        "--index-csv",
        type=Path,
        default=Path("data/merl_index.csv"),
        help="Path to the MERL index CSV. Default: %(default)s",
    )
    parser.add_argument(
        "--background-h5",
        type=Path,
        default=Path("data/MERLGPR/MERLGPR/data/v1/data_frequency_freespace.h5"),
        help="Path to the freespace/background HDF5 file. Default: %(default)s",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to inspect. Default: %(default)s",
    )
    parser.add_argument(
        "--input-mode",
        default="magnitude",
        choices=sorted(VALID_INPUT_MODES),
        help="Input representation mode. Default: %(default)s",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=4,
        help="Number of samples to inspect. Default: %(default)s",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/debug"),
        help="Directory for debug visualizations. Default: %(default)s",
    )
    return parser.parse_args()


def print_header(title: str) -> None:
    print(f"\n{'=' * 80}")
    print(title)
    print("=" * 80)


def summarize_tensor(name: str, tensor: torch.Tensor) -> None:
    tensor_cpu = tensor.detach().cpu().float()
    print(
        f"{name}: shape={tuple(tensor_cpu.shape)}, dtype={tensor_cpu.dtype}, "
        f"min={float(tensor_cpu.min()):.6g}, max={float(tensor_cpu.max()):.6g}, "
        f"mean={float(tensor_cpu.mean()):.6g}"
    )


def ensure_output_dir(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def save_magnitude_visualization(
    sample_id: str,
    input_tensor: torch.Tensor,
    target_tensor: torch.Tensor,
    output_dir: Path,
) -> Path:
    input_np = input_tensor.detach().cpu().numpy()
    target_np = target_tensor.detach().cpu().numpy()

    first_frequency_slice = input_np[0]
    mean_over_frequencies = input_np.mean(axis=0)
    target_eps = target_np[0]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)

    panels = [
        ("Input: frequency slice 0", first_frequency_slice, "viridis"),
        ("Input: mean over frequencies", mean_over_frequencies, "viridis"),
        ("Target: eps", target_eps, "magma"),
    ]

    for axis, (title, image, cmap) in zip(axes, panels):
        im = axis.imshow(image, cmap=cmap)
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
        fig.colorbar(im, ax=axis, fraction=0.046, pad=0.04)

    output_path = output_dir / f"sample_{sample_id}_magnitude_debug.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def main() -> int:
    args = parse_args()
    if args.num_samples <= 0:
        print("num_samples must be > 0.", file=sys.stderr)
        return 1

    output_dir = ensure_output_dir(args.output_dir)

    print_header("Debug Configuration")
    print(f"index_csv={args.index_csv.expanduser().resolve()}")
    print(f"background_h5={args.background_h5.expanduser().resolve()}")
    print(f"split={args.split}")
    print(f"input_mode={args.input_mode}")
    print(f"num_samples={args.num_samples}")
    print(f"output_dir={output_dir}")

    try:
        dataset = MerlDataset(
            index_csv=args.index_csv,
            background_h5=args.background_h5,
            split=args.split,
            input_mode=args.input_mode,
            normalize_input=True,
        )
    except Exception as exc:
        print(f"Failed to initialize MerlDataset: {exc}", file=sys.stderr)
        return 1

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    print_header("Dataset Summary")
    print(f"dataset_length={len(dataset)}")

    saved_paths: list[Path] = []
    try:
        for batch_index, batch in enumerate(loader):
            if batch_index >= args.num_samples:
                break

            sample_id = str(batch["sample_id"][0])
            input_tensor = batch["input"][0]
            target_tensor = batch["target"][0]

            print_header(f"Sample {batch_index + 1}: sample_id={sample_id}")
            summarize_tensor("input", input_tensor)
            summarize_tensor("target", target_tensor)

            if args.input_mode == "magnitude":
                output_path = save_magnitude_visualization(
                    sample_id=sample_id,
                    input_tensor=input_tensor,
                    target_tensor=target_tensor,
                    output_dir=output_dir,
                )
                saved_paths.append(output_path)
                print(f"saved_visualization={output_path}")
            else:
                print(
                    "Visualization skipped because the requested layout is only defined for input_mode='magnitude'."
                )
    except Exception as exc:
        print(f"Failed while iterating over the DataLoader: {exc}", file=sys.stderr)
        dataset.close()
        return 1

    dataset.close()

    print_header("Done")
    if saved_paths:
        for path in saved_paths:
            print(path)
    else:
        print("No visualization files were written.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
