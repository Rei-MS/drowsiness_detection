"""
SequenceDataset — ventanas temporales de N frames consecutivos del mismo video.

Detecta runs contiguos (mismo sujeto + mismo label + índices h5 consecutivos)
y genera ventanas de seq_len frames. Cada ítem: (seq_len, C, H, W) + label.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.dataset import build_transforms


def _build_windows(
    h5_path: str,
    frame_indices: np.ndarray,
    seq_len: int,
    seq_stride: int,
    label_map: Optional[dict],
) -> list[tuple[list[int], int]]:
    """
    Retorna lista de (positions_in_frame_indices, label) para cada ventana válida.
    Un run termina cuando cambia el sujeto, el label, o hay un salto en el índice h5.
    """
    with h5py.File(h5_path, "r") as f:
        y        = f["y"][frame_indices]
        subjects = f["subjects"][frame_indices].astype(str)

    if label_map is not None:
        y_mapped = np.array([label_map.get(int(v), -1) for v in y])
    else:
        y_mapped = y.astype(int)

    windows: list[tuple[list[int], int]] = []
    n = len(frame_indices)
    i = 0

    while i < n:
        j = i + 1
        while (
            j < n
            and subjects[j] == subjects[i]
            and y[j] == y[i]
            and int(frame_indices[j]) == int(frame_indices[j - 1]) + 1
        ):
            j += 1

        run_len = j - i
        label   = int(y_mapped[i])

        if label >= 0 and run_len >= seq_len:
            for start in range(0, run_len - seq_len + 1, seq_stride):
                windows.append((list(range(i + start, i + start + seq_len)), label))
        i = j

    return windows


class SequenceDataset(Dataset):
    def __init__(
        self,
        h5_path: str | Path,
        frame_indices: np.ndarray,
        seq_len: int = 16,
        seq_stride: int = 8,
        transform=None,
        label_map: Optional[dict] = None,
    ):
        self.h5_path      = str(h5_path)
        self.frame_indices = frame_indices
        self.seq_len      = seq_len
        self.transform    = transform
        self._file: Optional[h5py.File] = None

        self.windows = _build_windows(
            self.h5_path, frame_indices, seq_len, seq_stride, label_map
        )

    def __len__(self) -> int:
        return len(self.windows)

    def _open(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r")
        return self._file

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        positions, label = self.windows[idx]
        h5_idxs = [int(self.frame_indices[p]) for p in positions]

        frames = self._open()["X"][h5_idxs]   # (T, H, W, C) uint8

        if self.transform is not None:
            seq = torch.stack([self.transform(image=frames[t])["image"]
                               for t in range(self.seq_len)])   # (T, C, H, W)
        else:
            seq = torch.from_numpy(frames.transpose(0, 3, 1, 2)).float() / 255.0

        return seq, label

    def __del__(self):
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass


def make_sequence_loaders(
    h5_path: str | Path,
    train_idx: np.ndarray,
    val_idx:   np.ndarray,
    test_idx:  np.ndarray,
    seq_len:      int = 16,
    train_stride: int = 4,    # overlapping → más muestras de entrenamiento
    eval_stride:  int = 16,   # non-overlapping para val/test
    batch_size:   int = 32,
    num_workers:  int = 8,
    label_map:    Optional[dict] = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = SequenceDataset(h5_path, train_idx, seq_len, train_stride,
                               build_transforms(augment=True),  label_map)
    val_ds   = SequenceDataset(h5_path, val_idx,   seq_len, eval_stride,
                               build_transforms(augment=False), label_map)
    test_ds  = SequenceDataset(h5_path, test_idx,  seq_len, eval_stride,
                               build_transforms(augment=False), label_map)

    persistent = num_workers > 0
    kw = dict(pin_memory=True, persistent_workers=persistent,
              prefetch_factor=2 if persistent else None)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, **kw)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, **kw)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, **kw)

    return train_loader, val_loader, test_loader
