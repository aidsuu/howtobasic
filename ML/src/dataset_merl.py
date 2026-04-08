#!/usr/bin/env python3
"""PyTorch dataset utilities for the MERL-GPR inversion dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


VALID_INPUT_MODES = {
    "magnitude",
    "real_imag",
    "magnitude_minus_freespace",
}
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class MerlSampleRecord:
    sample_id: str
    h5_path: Path
    group_name: str
    target_key: str
    input_key: str
    split: str


class MerlDataset(Dataset):
    """Read MERL-GPR samples from the prebuilt CSV index.

    The dataset is intentionally simple and safe:
    - HDF5 files are opened lazily and read in read-only mode.
    - Inputs are converted to float32 and normalized per sample.
    - Targets are kept real-valued float32 and are not normalized.
    - Metadata includes sample_id for traceability during debugging.
    """

    def __init__(
        self,
        index_csv: str | Path = "data/merl_index.csv",
        background_h5: str | Path = "data/MERLGPR/MERLGPR/data/v1/data_frequency_freespace.h5",
        split: Optional[str] = None,
        input_mode: str = "magnitude",
        normalize_input: bool = True,
        normalization_eps: float = 1e-6,
    ) -> None:
        self.index_csv = self._resolve_path(index_csv)
        self.background_h5_path = self._resolve_path(background_h5)
        self.split = split
        self.input_mode = input_mode
        self.normalize_input = normalize_input
        self.normalization_eps = float(normalization_eps)
        self._h5_handles: Dict[Path, h5py.File] = {}
        self._background_cache: Optional[Dict[str, np.ndarray]] = None

        self._validate_init_args()
        self.records = self._load_records()

    @staticmethod
    def _resolve_path(path_like: str | Path) -> Path:
        path = Path(path_like).expanduser()
        if path.is_absolute():
            return path.resolve()
        return (PROJECT_ROOT / path).resolve()

    def _validate_init_args(self) -> None:
        if self.input_mode not in VALID_INPUT_MODES:
            raise ValueError(
                f"Unsupported input_mode={self.input_mode!r}. "
                f"Expected one of {sorted(VALID_INPUT_MODES)}."
            )

        if not self.index_csv.exists():
            raise FileNotFoundError(f"Index CSV not found: {self.index_csv}")
        if not self.index_csv.is_file():
            raise FileNotFoundError(f"Index CSV is not a file: {self.index_csv}")

        if self.normalization_eps <= 0:
            raise ValueError("normalization_eps must be > 0.")

    def _load_records(self) -> list[MerlSampleRecord]:
        try:
            frame = pd.read_csv(self.index_csv, dtype=str)
        except Exception as exc:
            raise RuntimeError(f"Failed to read index CSV {self.index_csv}: {exc}") from exc

        required_columns = {
            "sample_id",
            "h5_path",
            "group_name",
            "target_key",
            "input_key",
            "split",
        }
        missing_columns = required_columns.difference(frame.columns)
        if missing_columns:
            raise ValueError(
                f"Index CSV is missing required columns: {sorted(missing_columns)}"
            )

        if self.split is not None:
            frame = frame[frame["split"] == self.split]

        if frame.empty:
            split_note = f" for split={self.split!r}" if self.split is not None else ""
            raise ValueError(f"No dataset rows found in {self.index_csv}{split_note}.")

        records: list[MerlSampleRecord] = []
        for row in frame.to_dict(orient="records"):
            h5_path = Path(row["h5_path"]).expanduser().resolve()
            if not h5_path.exists():
                raise FileNotFoundError(
                    f"HDF5 file referenced by index does not exist: {h5_path}"
                )
            records.append(
                MerlSampleRecord(
                    sample_id=str(row["sample_id"]),
                    h5_path=h5_path,
                    group_name=str(row["group_name"]),
                    target_key=str(row["target_key"]),
                    input_key=str(row["input_key"]),
                    split=str(row["split"]),
                )
            )

        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        try:
            input_array, target_array = self._read_sample(record)
            input_array = self._build_input_tensor(input_array)
            if self.normalize_input:
                input_array = self._normalize_per_sample(input_array)

            target_array = np.asarray(target_array, dtype=np.float32)
            target_array = np.expand_dims(target_array, axis=0)

            return {
                "input": torch.from_numpy(np.ascontiguousarray(input_array)),
                "target": torch.from_numpy(np.ascontiguousarray(target_array)),
                "sample_id": record.sample_id,
                "metadata": {
                    "sample_id": record.sample_id,
                    "group_name": record.group_name,
                    "split": record.split,
                    "h5_path": str(record.h5_path),
                    "input_mode": self.input_mode,
                },
            }
        except Exception as exc:
            raise RuntimeError(
                "Failed to load MERL sample "
                f"sample_id={record.sample_id}, group_name={record.group_name}, "
                f"h5_path={record.h5_path}: {exc}"
            ) from exc

    def _get_h5_handle(self, path: Path) -> h5py.File:
        handle = self._h5_handles.get(path)
        if handle is None:
            handle = h5py.File(path, "r")
            self._h5_handles[path] = handle
        return handle

    def _read_sample(self, record: MerlSampleRecord) -> tuple[np.ndarray, np.ndarray]:
        handle = self._get_h5_handle(record.h5_path)
        if record.group_name not in handle:
            raise KeyError(f"Group {record.group_name!r} not found in HDF5 file.")

        group = handle[record.group_name]
        if record.input_key not in group:
            raise KeyError(f"Input key {record.input_key!r} not found in group {record.group_name!r}.")
        if record.target_key not in group:
            raise KeyError(f"Target key {record.target_key!r} not found in group {record.group_name!r}.")

        input_array = group[record.input_key][()]
        target_array = group[record.target_key][()]

        if input_array.shape != (50, 63, 63):
            raise ValueError(
                f"Unexpected input shape {input_array.shape}; expected (50, 63, 63)."
            )
        if target_array.shape != (63, 63):
            raise ValueError(
                f"Unexpected target shape {target_array.shape}; expected (63, 63)."
            )

        return input_array, target_array

    def _load_background(self) -> Optional[Dict[str, np.ndarray]]:
        if self._background_cache is not None:
            return self._background_cache

        if not self.background_h5_path.exists():
            self._background_cache = None
            return None

        if not self.background_h5_path.is_file():
            raise FileNotFoundError(
                f"Background HDF5 path is not a file: {self.background_h5_path}"
            )

        with h5py.File(self.background_h5_path, "r") as handle:
            if "f_field" not in handle:
                raise KeyError(
                    f"Background file does not contain top-level dataset 'f_field': {self.background_h5_path}"
                )

            background_f_field = handle["f_field"][()]
            if background_f_field.shape != (50, 63, 63):
                raise ValueError(
                    "Unexpected background f_field shape "
                    f"{background_f_field.shape}; expected (50, 63, 63)."
                )

            payload: Dict[str, np.ndarray] = {
                "f_field": background_f_field,
            }
            if "frequency" in handle:
                payload["frequency"] = handle["frequency"][()]

        self._background_cache = payload
        return self._background_cache

    def _build_input_tensor(self, f_field: np.ndarray) -> np.ndarray:
        if self.input_mode == "magnitude":
            output = np.abs(f_field).astype(np.float32, copy=False)
        elif self.input_mode == "real_imag":
            real = np.real(f_field).astype(np.float32, copy=False)
            imag = np.imag(f_field).astype(np.float32, copy=False)
            output = np.concatenate([real, imag], axis=0)
        elif self.input_mode == "magnitude_minus_freespace":
            background = self._load_background()
            if background is None:
                raise FileNotFoundError(
                    "Input mode 'magnitude_minus_freespace' requires the freespace HDF5 file, "
                    f"but it was not found at {self.background_h5_path}."
                )
            background_f_field = background["f_field"]
            output = (
                np.abs(f_field).astype(np.float32, copy=False)
                - np.abs(background_f_field).astype(np.float32, copy=False)
            )
        else:
            raise ValueError(
                f"Unsupported input_mode={self.input_mode!r}. "
                f"Expected one of {sorted(VALID_INPUT_MODES)}."
            )

        return np.asarray(output, dtype=np.float32)

    def _normalize_per_sample(self, array: np.ndarray) -> np.ndarray:
        mean = float(np.mean(array))
        std = float(np.std(array))

        if not np.isfinite(mean) or not np.isfinite(std):
            raise ValueError(
                f"Non-finite statistics encountered during normalization: mean={mean}, std={std}."
            )

        if std < self.normalization_eps:
            max_abs = float(np.max(np.abs(array)))
            if max_abs < self.normalization_eps:
                return np.zeros_like(array, dtype=np.float32)
            return np.asarray(array / max_abs, dtype=np.float32)

        return np.asarray((array - mean) / std, dtype=np.float32)

    def close(self) -> None:
        for handle in self._h5_handles.values():
            try:
                handle.close()
            except Exception:
                pass
        self._h5_handles.clear()

    def __del__(self) -> None:
        self.close()
