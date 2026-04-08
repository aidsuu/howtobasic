#!/usr/bin/env python3
"""Verify MERL-GPR HDF5 structure in read-only mode."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import h5py
import numpy as np


DEFAULT_MAIN_H5 = Path("data/MERLGPR/MERLGPR/data/v1/data_frequency.h5")
DEFAULT_BACKGROUND_H5 = Path("data/MERLGPR/MERLGPR/data/v1/data_frequency_freespace.h5")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify MERL-GPR sample structure and background layout without modifying source files."
    )
    parser.add_argument(
        "--main-h5",
        type=Path,
        default=DEFAULT_MAIN_H5,
        help=f"Path to the main sample HDF5 file. Default: {DEFAULT_MAIN_H5}",
    )
    parser.add_argument(
        "--background-h5",
        type=Path,
        default=DEFAULT_BACKGROUND_H5,
        help=f"Path to the freespace/background HDF5 file. Default: {DEFAULT_BACKGROUND_H5}",
    )
    parser.add_argument(
        "--preview-count",
        type=int,
        default=5,
        help="Number of sample groups to preview. Default: %(default)s",
    )
    return parser.parse_args()


def print_header(title: str) -> None:
    print(f"\n{'=' * 80}")
    print(title)
    print("=" * 80)


def numeric_sort_key(name: str) -> Tuple[int, object]:
    return (0, int(name)) if name.isdigit() else (1, name)


def summarize_eps(array: np.ndarray) -> str:
    return (
        f"shape={array.shape}, dtype={array.dtype}, "
        f"min={float(np.min(array)):.6g}, max={float(np.max(array)):.6g}"
    )


def summarize_complex_magnitude(array: np.ndarray) -> str:
    magnitude = np.abs(array)
    return (
        f"shape={array.shape}, dtype={array.dtype}, complex_dtype={np.iscomplexobj(array)}, "
        f"magnitude_min={float(np.min(magnitude)):.6g}, magnitude_max={float(np.max(magnitude)):.6g}"
    )


def check_path(path: Path, description: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{description} not found: {resolved}")
    if not resolved.is_file():
        raise FileNotFoundError(f"{description} is not a file: {resolved}")
    return resolved


def get_numeric_group_names(handle: h5py.File) -> List[str]:
    return sorted([name for name in handle.keys() if name.isdigit()], key=numeric_sort_key)


def verify_group_layout(handle: h5py.File, group_names: Sequence[str]) -> Tuple[Tuple[str, ...] | None, List[Tuple[str, Tuple[str, ...]]]]:
    expected_keys: Tuple[str, ...] | None = None
    mismatches: List[Tuple[str, Tuple[str, ...]]] = []

    for name in group_names:
        keys = tuple(sorted(handle[name].keys()))
        if expected_keys is None:
            expected_keys = keys
            continue
        if keys != expected_keys:
            mismatches.append((name, keys))

    return expected_keys, mismatches


def print_assumptions(main_h5: Path, background_h5: Path, group_names: Sequence[str], expected_keys: Tuple[str, ...] | None) -> None:
    print_header("Explicit Assumptions")
    print(f"Assumption 1: sample groups are numeric HDF5 groups in {main_h5}.")
    print(f"Observed: found {len(group_names)} numeric groups.")
    print("Assumption 2: input_key and target_key come from the observed group keys, not from external metadata.")
    print(f"Observed: common keys across numeric groups = {expected_keys}.")
    print(
        f"Assumption 3: {background_h5} is treated as a single global background only if it has one shared dataset layout "
        "instead of per-sample numeric groups."
    )


def print_sample_preview(handle: h5py.File, group_names: Sequence[str], preview_count: int) -> None:
    print_header(f"First {min(preview_count, len(group_names))} Sample Groups")
    for name in group_names[:preview_count]:
        group = handle[name]
        eps = group["eps"][()]
        f_field = group["f_field"][()]
        print(f"sample_id={name}")
        print(f"  keys={tuple(sorted(group.keys()))}")
        print(f"  eps: {summarize_eps(eps)}")
        print(f"  f_field: {summarize_complex_magnitude(f_field)}")


def verify_background_file(background_h5: Path, reference_shape: Tuple[int, ...] | None) -> None:
    print_header("Background File Verification")
    with h5py.File(background_h5, "r") as handle:
        top_level_keys = list(handle.keys())
        numeric_groups = [name for name in top_level_keys if name.isdigit()]

        print(f"Background file: {background_h5}")
        print(f"Top-level keys: {top_level_keys}")
        print(f"Numeric groups found: {numeric_groups}")

        if "f_field" in handle:
            background_f_field = handle["f_field"][()]
            print(f"background f_field: {summarize_complex_magnitude(background_f_field)}")
            if reference_shape is not None:
                print(
                    "Observed comparison: "
                    f"background f_field shape matches sample f_field shape = {background_f_field.shape == reference_shape}"
                )
        else:
            print("Observed: background file does not contain a top-level f_field dataset.")

        if "frequency" in handle:
            frequency = handle["frequency"][()]
            print(
                "background frequency: "
                f"shape={frequency.shape}, dtype={frequency.dtype}, "
                f"min={float(np.min(frequency)):.6g}, max={float(np.max(frequency)):.6g}"
            )
        else:
            print("Observed: background file does not contain a top-level frequency dataset.")

        if numeric_groups:
            print(
                "Conclusion: background file is not a single global background because it contains numeric per-sample groups."
            )
        else:
            print(
                "Conclusion: background file contains one shared top-level layout and no numeric per-sample groups."
            )
            print(
                "Explicit assumption: based on file structure alone, this supports treating the background as global/shared "
                "across all samples."
            )


def main() -> int:
    args = parse_args()

    try:
        main_h5 = check_path(args.main_h5, "Main HDF5 file")
        background_h5 = check_path(args.background_h5, "Background HDF5 file")
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    with h5py.File(main_h5, "r") as handle:
        group_names = get_numeric_group_names(handle)
        if not group_names:
            print("No numeric sample groups were found in the main HDF5 file.", file=sys.stderr)
            return 1

        expected_keys, mismatches = verify_group_layout(handle, group_names)
        sample_shape = tuple(handle[group_names[0]]["f_field"].shape) if "f_field" in handle[group_names[0]] else None

        print_assumptions(main_h5, background_h5, group_names, expected_keys)

        print_header("Group Layout Verification")
        print(f"Main file: {main_h5}")
        print(f"Numeric groups found: {len(group_names)}")
        print(f"Common keys in numeric groups: {expected_keys}")
        print(f"Groups with mismatched keys: {len(mismatches)}")
        if mismatches:
            for name, keys in mismatches[:10]:
                print(f"  mismatch group={name}, keys={keys}")
        else:
            print("All numeric groups expose the same keys.")

        print_sample_preview(handle, group_names, preview_count=args.preview_count)

    verify_background_file(background_h5, reference_shape=sample_shape)

    print("\nVerification completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
