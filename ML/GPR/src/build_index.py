#!/usr/bin/env python3
"""Build a reproducible CSV index for MERL-GPR samples."""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import h5py


DEFAULT_MAIN_H5 = Path("data/MERLGPR/MERLGPR/data/v1/data_frequency.h5")
DEFAULT_OUTPUT_CSV = Path("data/merl_index.csv")
DEFAULT_SEED = 20260407
DEFAULT_SPLITS = (0.70, 0.15, 0.15)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a reproducible CSV index from numeric MERL-GPR HDF5 sample groups."
    )
    parser.add_argument(
        "--main-h5",
        type=Path,
        default=DEFAULT_MAIN_H5,
        help=f"Path to the main HDF5 dataset. Default: {DEFAULT_MAIN_H5}",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help=f"Output CSV path. Default: {DEFAULT_OUTPUT_CSV}",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Fixed random seed for reproducible splits. Default: %(default)s",
    )
    return parser.parse_args()


def print_header(title: str) -> None:
    print(f"\n{'=' * 80}")
    print(title)
    print("=" * 80)


def check_path(path: Path, description: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{description} not found: {resolved}")
    if not resolved.is_file():
        raise FileNotFoundError(f"{description} is not a file: {resolved}")
    return resolved


def numeric_sort_key(name: str) -> Tuple[int, object]:
    return (0, int(name)) if name.isdigit() else (1, name)


def get_numeric_group_names(handle: h5py.File) -> List[str]:
    return sorted([name for name in handle.keys() if name.isdigit()], key=numeric_sort_key)


def verify_group_keys(handle: h5py.File, group_names: Sequence[str]) -> Tuple[str, str]:
    if not group_names:
        raise ValueError("No numeric groups found in the HDF5 file.")

    expected_keys = tuple(sorted(handle[group_names[0]].keys()))
    if expected_keys != ("eps", "f_field"):
        raise ValueError(
            "Observed group keys do not match the expected MERL layout ('eps', 'f_field'). "
            f"Observed keys: {expected_keys}"
        )

    for name in group_names[1:]:
        current_keys = tuple(sorted(handle[name].keys()))
        if current_keys != expected_keys:
            raise ValueError(f"Group {name} has inconsistent keys: {current_keys}")

    # Explicit mapping derived from observed keys in the HDF5 structure.
    target_key = "eps"
    input_key = "f_field"
    return target_key, input_key


def assign_splits(group_names: Sequence[str], seed: int) -> List[Tuple[str, str]]:
    names = list(group_names)
    rng = random.Random(seed)
    shuffled = names[:]
    rng.shuffle(shuffled)

    total = len(shuffled)
    train_count = int(total * DEFAULT_SPLITS[0])
    val_count = int(total * DEFAULT_SPLITS[1])
    test_count = total - train_count - val_count

    split_by_name = {}
    for name in shuffled[:train_count]:
        split_by_name[name] = "train"
    for name in shuffled[train_count:train_count + val_count]:
        split_by_name[name] = "val"
    for name in shuffled[train_count + val_count:]:
        split_by_name[name] = "test"

    counts = {"train": train_count, "val": val_count, "test": test_count}
    print(
        "Split counts derived from total samples and fixed ratios "
        f"{DEFAULT_SPLITS}: {counts}"
    )
    return [(name, split_by_name[name]) for name in names]


def write_index_csv(
    output_csv: Path,
    main_h5: Path,
    rows: Sequence[Tuple[str, str]],
    target_key: str,
    input_key: str,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    project_root = output_csv.parent.parent.resolve()
    try:
        portable_h5_path = main_h5.relative_to(project_root)
    except ValueError:
        portable_h5_path = main_h5
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_id", "h5_path", "group_name", "target_key", "input_key", "split"],
        )
        writer.writeheader()
        for sample_id, split in rows:
            writer.writerow(
                {
                    "sample_id": sample_id,
                    "h5_path": str(portable_h5_path),
                    "group_name": sample_id,
                    "target_key": target_key,
                    "input_key": input_key,
                    "split": split,
                }
            )


def main() -> int:
    args = parse_args()

    try:
        main_h5 = check_path(args.main_h5, "Main HDF5 file")
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    output_csv = args.output_csv.expanduser().resolve()

    print_header("Explicit Assumptions")
    print(f"Assumption 1: numeric top-level groups in {main_h5} represent samples.")
    print("Assumption 2: target_key and input_key are inferred only from observed group keys.")
    print(
        f"Assumption 3: split ratios are fixed at {DEFAULT_SPLITS} with seed={args.seed} for reproducibility."
    )
    print(
        f"Assumption 4: output index CSV may be created at {output_csv}, but source HDF5 data will remain unchanged."
    )

    try:
        with h5py.File(main_h5, "r") as handle:
            group_names = get_numeric_group_names(handle)
            target_key, input_key = verify_group_keys(handle, group_names)
    except Exception as exc:
        print(f"Failed to inspect HDF5 structure: {exc}", file=sys.stderr)
        return 1

    print_header("Observed Structure")
    print(f"Numeric sample groups found: {len(group_names)}")
    print(f"Observed target_key: {target_key}")
    print(f"Observed input_key: {input_key}")
    print(f"First 5 sample ids: {group_names[:5]}")

    rows = assign_splits(group_names, seed=args.seed)

    try:
        write_index_csv(
            output_csv=output_csv,
            main_h5=main_h5,
            rows=rows,
            target_key=target_key,
            input_key=input_key,
        )
    except Exception as exc:
        print(f"Failed to write CSV index: {exc}", file=sys.stderr)
        return 1

    print_header("Index Output")
    print(f"CSV written to: {output_csv}")
    print("Preview rows:")
    for sample_id, split in rows[:5]:
        print(
            f"  sample_id={sample_id}, h5_path={main_h5}, group_name={sample_id}, "
            f"target_key={target_key}, input_key={input_key}, split={split}"
        )

    print("\nIndex build completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
