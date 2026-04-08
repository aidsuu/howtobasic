#!/usr/bin/env python3
"""Dataset inspection utilities for the MERL-GPR project.

This script is read-only. It traverses a dataset directory, prints a compact
directory tree, summarizes file types, and inspects common scientific data
formats such as HDF5, MAT, NPY, and NPZ.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import h5py
import numpy as np
from scipy.io import loadmat, whosmat


DATA_EXTENSIONS = {
    ".h5",
    ".hdf5",
    ".mat",
    ".npy",
    ".npz",
    ".csv",
    ".txt",
    ".tsv",
    ".json",
}

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".gif",
    ".tif",
    ".tiff",
    ".webp",
}

RELEVANT_EXTENSIONS = DATA_EXTENSIONS | IMAGE_EXTENSIONS


@dataclass
class FileSummary:
    path: Path
    extension: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect a dataset folder without modifying the original data."
    )
    parser.add_argument(
        "dataset_root",
        nargs="?",
        default="data/MERLGPR",
        help="Path to the dataset root directory. Default: %(default)s",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=4,
        help="Maximum depth shown in the compact directory tree. Default: %(default)s",
    )
    parser.add_argument(
        "--max-entries-per-dir",
        type=int,
        default=8,
        help="Maximum child entries shown per directory in the compact tree. Default: %(default)s",
    )
    parser.add_argument(
        "--max-samples-per-ext",
        type=int,
        default=5,
        help="Maximum example file names shown for each extension. Default: %(default)s",
    )
    parser.add_argument(
        "--max-array-items",
        type=int,
        default=40,
        help="Maximum array or dataset entries shown per inspected file. Default: %(default)s",
    )
    return parser.parse_args()


def normalize_extension(path: Path) -> str:
    return path.suffix.lower() if path.suffix else "<no_ext>"


def safe_relpath(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def find_relevant_files(root: Path) -> List[FileSummary]:
    results: List[FileSummary] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        ext = normalize_extension(path)
        if ext in RELEVANT_EXTENSIONS:
            results.append(FileSummary(path=path, extension=ext))
    return results


def print_header(title: str) -> None:
    print(f"\n{'=' * 80}")
    print(title)
    print("=" * 80)


def build_tree_lines(
    root: Path,
    max_depth: int,
    max_entries_per_dir: int,
) -> List[str]:
    lines = [f"{root.name}/"]

    def walk(directory: Path, prefix: str, depth: int) -> None:
        if depth >= max_depth:
            return

        try:
            children = sorted(directory.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except OSError as exc:
            lines.append(f"{prefix}[error reading directory: {exc}]")
            return

        shown_children = children[:max_entries_per_dir]
        hidden_count = len(children) - len(shown_children)

        for index, child in enumerate(shown_children):
            is_last_visible = index == len(shown_children) - 1 and hidden_count == 0
            branch = "└── " if is_last_visible else "├── "
            suffix = "/" if child.is_dir() else ""
            lines.append(f"{prefix}{branch}{child.name}{suffix}")

            if child.is_dir():
                next_prefix = prefix + ("    " if is_last_visible else "│   ")
                walk(child, next_prefix, depth + 1)

        if hidden_count > 0:
            lines.append(f"{prefix}└── ... ({hidden_count} more entries)")

    walk(root, prefix="", depth=0)
    return lines


def print_tree(root: Path, max_depth: int, max_entries_per_dir: int) -> None:
    print_header("Compact Folder Tree")
    for line in build_tree_lines(root, max_depth=max_depth, max_entries_per_dir=max_entries_per_dir):
        print(line)


def summarize_extensions(files: Sequence[FileSummary], max_samples_per_ext: int, root: Path) -> None:
    print_header("Relevant File Counts")

    counts = Counter(file.extension for file in files)
    examples: Dict[str, List[str]] = defaultdict(list)

    for file in files:
        ext = file.extension
        if len(examples[ext]) < max_samples_per_ext:
            examples[ext].append(safe_relpath(file.path, root))

    if not counts:
        print("No relevant dataset files were found.")
        return

    for ext, count in sorted(counts.items(), key=lambda item: (item[0] == "<no_ext>", item[0])):
        print(f"{ext}: {count}")
        for sample in examples[ext]:
            print(f"  - {sample}")


def inspect_hdf5_file(path: Path, root: Path, max_items: int) -> None:
    print(f"\n[HDF5] {safe_relpath(path, root)}")
    try:
        with h5py.File(path, "r") as handle:
            entries: List[str] = []
            total_items = 0

            def visitor(name: str, obj: h5py.HLObject) -> None:
                nonlocal total_items
                total_items += 1
                if len(entries) >= max_items:
                    return
                if isinstance(obj, h5py.Group):
                    entries.append(f"  GROUP   {name or '/'}")
                elif isinstance(obj, h5py.Dataset):
                    entries.append(
                        f"  DATASET {name}: shape={obj.shape}, dtype={obj.dtype}"
                    )

            handle.visititems(visitor)

            if not entries:
                print("  Empty HDF5 file.")
                return

            for line in entries:
                print(line)
    except Exception as exc:
        print(f"  Error reading HDF5 file: {exc}")
        return

    if total_items > max_items:
        print(f"  ... output truncated after {max_items} items (total discovered: {total_items})")


def inspect_mat_file(path: Path, root: Path, max_items: int) -> None:
    print(f"\n[MAT] {safe_relpath(path, root)}")
    try:
        variables = whosmat(path)
        if variables:
            for name, shape, dtype in variables[:max_items]:
                print(f"  {name}: shape={shape}, dtype={dtype}")
            if len(variables) > max_items:
                print(f"  ... output truncated after {max_items} arrays")
            return

        # Fallback for MAT files where whosmat is insufficient.
        content = loadmat(path)
        shown = 0
        for name, value in sorted(content.items()):
            if name.startswith("__"):
                continue
            if shown >= max_items:
                break
            shape = getattr(value, "shape", None)
            dtype = getattr(value, "dtype", type(value).__name__)
            print(f"  {name}: shape={shape}, dtype={dtype}")
            shown += 1
        if shown == 0:
            print("  No user arrays found in MAT file.")
    except Exception as exc:
        print(f"  Error reading MAT file: {exc}")


def inspect_npy_file(path: Path, root: Path) -> None:
    print(f"\n[NPY] {safe_relpath(path, root)}")
    try:
        try:
            array = np.load(path, allow_pickle=False, mmap_mode="r")
        except ValueError:
            array = np.load(path, allow_pickle=True)

        if isinstance(array, np.ndarray):
            print(f"  array: shape={array.shape}, dtype={array.dtype}")
        else:
            print(f"  object: type={type(array).__name__}")
    except Exception as exc:
        print(f"  Error reading NPY file: {exc}")


def inspect_npz_file(path: Path, root: Path, max_items: int) -> None:
    print(f"\n[NPZ] {safe_relpath(path, root)}")
    try:
        with np.load(path, allow_pickle=False) as archive:
            names = archive.files
            if not names:
                print("  Empty NPZ archive.")
                return
            for name in names[:max_items]:
                value = archive[name]
                print(f"  {name}: shape={value.shape}, dtype={value.dtype}")
            if len(names) > max_items:
                print(f"  ... output truncated after {max_items} arrays")
    except ValueError:
        try:
            with np.load(path, allow_pickle=True) as archive:
                names = archive.files
                for name in names[:max_items]:
                    value = archive[name]
                    shape = getattr(value, "shape", None)
                    dtype = getattr(value, "dtype", type(value).__name__)
                    print(f"  {name}: shape={shape}, dtype={dtype}")
                if len(names) > max_items:
                    print(f"  ... output truncated after {max_items} arrays")
        except Exception as exc:
            print(f"  Error reading NPZ file: {exc}")
    except Exception as exc:
        print(f"  Error reading NPZ file: {exc}")


def inspect_array_files(files: Sequence[FileSummary], root: Path, max_items: int) -> None:
    print_header("Array And Container Inspection")

    inspected_any = False
    for file in files:
        if file.extension in {".h5", ".hdf5"}:
            inspected_any = True
            inspect_hdf5_file(file.path, root=root, max_items=max_items)
        elif file.extension == ".mat":
            inspected_any = True
            inspect_mat_file(file.path, root=root, max_items=max_items)
        elif file.extension == ".npy":
            inspected_any = True
            inspect_npy_file(file.path, root=root)
        elif file.extension == ".npz":
            inspected_any = True
            inspect_npz_file(file.path, root=root, max_items=max_items)

    if not inspected_any:
        print("No HDF5, MAT, NPY, or NPZ files were found.")


def print_plain_file_overview(files: Sequence[FileSummary], root: Path) -> None:
    print_header("Text And Image Files")

    plain_files = [
        file for file in files if file.extension in (DATA_EXTENSIONS | IMAGE_EXTENSIONS) - {".h5", ".hdf5", ".mat", ".npy", ".npz"}
    ]

    if not plain_files:
        print("No text or image files matched the relevant extension list.")
        return

    for file in plain_files:
        print(f"{file.extension}: {safe_relpath(file.path, root)}")


def run_inspection(
    dataset_root: Path,
    max_depth: int = 4,
    max_entries_per_dir: int = 8,
    max_samples_per_ext: int = 5,
    max_array_items: int = 40,
) -> int:
    dataset_root = dataset_root.expanduser().resolve()

    print_header("Dataset Inspection")
    print(f"Dataset root: {dataset_root}")

    if not dataset_root.exists():
        print("Error: dataset root does not exist.", file=sys.stderr)
        return 1
    if not dataset_root.is_dir():
        print("Error: dataset root is not a directory.", file=sys.stderr)
        return 1

    print_tree(dataset_root, max_depth=max_depth, max_entries_per_dir=max_entries_per_dir)

    files = find_relevant_files(dataset_root)
    summarize_extensions(files, max_samples_per_ext=max_samples_per_ext, root=dataset_root)
    inspect_array_files(files, root=dataset_root, max_items=max_array_items)
    print_plain_file_overview(files, root=dataset_root)

    print("\nInspection completed successfully.")
    return 0


def main() -> int:
    args = parse_args()
    return run_inspection(
        dataset_root=Path(args.dataset_root),
        max_depth=args.max_depth,
        max_entries_per_dir=args.max_entries_per_dir,
        max_samples_per_ext=args.max_samples_per_ext,
        max_array_items=args.max_array_items,
    )


if __name__ == "__main__":
    raise SystemExit(main())
