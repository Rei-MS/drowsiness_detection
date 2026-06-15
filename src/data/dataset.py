"""
DrowsinessDataset con lectura lazy desde HDF5.

El archivo crops.h5 puede ser mucho más grande que la RAM disponible; el Dataset
abre el archivo una vez por worker de DataLoader y lee un frame a la vez, sin
cargar el array X completo en memoria.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    _HAS_ALBUMENTATIONS = True
except ImportError:
    _HAS_ALBUMENTATIONS = False

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
CLASS_NAMES   = ["alerta", "baja_vigilancia", "somnoliento"]


# Fracción central que se conserva al recortar (quita margen/fondo del crop Haar).
CENTER_CROP_FRAC = 0.75


def build_transforms(augment: bool = False):
    """
    Transforms diseñados para forzar al modelo a usar señales faciales de
    somnolencia y NO atajos espurios de sesión (iluminación, fondo, ropa).

    - Zoom al centro (RandomResizedCrop / CenterCrop) → descarta fondo y pelo.
    - Augmentation de color fuerte + ToGray → destruye la pista de iluminación
      que separa los 3 videos de cada sujeto.
    - CoarseDropout → robustez a oclusión, evita fijarse en un único parche.
    """
    if not _HAS_ALBUMENTATIONS:
        raise ImportError("albumentations no instalado. Ejecutar: uv add albumentations")

    if augment:
        return A.Compose([
            A.RandomResizedCrop(size=(224, 224), scale=(0.5, 1.0), ratio=(0.8, 1.25), p=1.0),
            A.HorizontalFlip(p=0.5),
            # Color agresivo: ataca el atajo de iluminación/sesión
            A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.8),
            A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=20, p=0.5),
            A.ToGray(p=0.2),
            A.Affine(rotate=(-12, 12), translate_percent={"x": (-0.06, 0.06), "y": (-0.06, 0.06)},
                     scale=(0.9, 1.1), p=0.6),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(0.1, 0.2),
                            hole_width_range=(0.1, 0.2), p=0.3),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])
    else:
        # Val/test: mismo zoom central determinista para matchear el dominio de train
        return A.Compose([
            A.CenterCrop(height=int(224 * CENTER_CROP_FRAC), width=int(224 * CENTER_CROP_FRAC), p=1.0),
            A.Resize(224, 224),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])


class DrowsinessDataset(Dataset):
    """
    Dataset lazy sobre crops.h5.

    El archivo HDF5 se abre una vez por worker (en el primer __getitem__),
    de modo que DataLoader con num_workers>0 funciona sin conflictos.
    """

    def __init__(self, h5_path: str | Path, indices: np.ndarray, transform=None, label_map=None):
        self.h5_path = str(h5_path)
        self.indices = indices
        self.transform = transform
        self._file: Optional[h5py.File] = None

        # Las etiquetas son pequeñas — se cargan completas una vez
        with h5py.File(self.h5_path, "r") as f:
            y = f["y"][indices]
        # label_map opcional para remapear clases (p.ej. binario {0:0, 2:1})
        if label_map is not None:
            y = np.array([label_map[int(v)] for v in y], dtype=y.dtype)
        self.y = y

    def __len__(self) -> int:
        return len(self.indices)

    def _open(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r")
        return self._file

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        image = self._open()["X"][self.indices[idx]]   # (224, 224, 3) uint8
        label = int(self.y[idx])

        if self.transform is not None:
            image = self.transform(image=image)["image"]
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        return image, label

    def __del__(self):
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass


def make_splits(
    h5_path: str | Path,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    keep_classes=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split por sujeto con distribución estratificada por clase.
    Lee solo 'y' y 'subjects' del HDF5 (pequeños), nunca carga X.

    Args:
        keep_classes: si se pasa (p.ej. [0, 2]), filtra los índices a esas clases.
            El split por sujeto se calcula con TODAS las clases y luego se filtra,
            así la integridad por sujeto se mantiene.

    Retorna:
        (train_indices, val_indices, test_indices)  — arrays de índices enteros
    """
    with h5py.File(h5_path, "r") as f:
        y        = f["y"][:]
        subjects = f["subjects"][:].astype(str)

    rng = np.random.default_rng(seed)
    unique_subjects = np.unique(subjects)

    subj_labels = {
        subj: int(np.bincount(y[subjects == subj]).argmax())
        for subj in unique_subjects
    }

    by_class: dict[int, list] = {0: [], 1: [], 2: []}
    for subj, lbl in subj_labels.items():
        by_class[lbl].append(subj)

    test_subjects, val_subjects = set(), set()
    for lbl, subj_list in by_class.items():
        arr = np.array(subj_list)
        rng.shuffle(arr)
        n_test = max(1, int(len(arr) * test_ratio))
        n_val  = max(1, int(len(arr) * val_ratio))
        test_subjects.update(arr[:n_test].tolist())
        val_subjects.update(arr[n_test:n_test + n_val].tolist())

    all_idx     = np.arange(len(y))
    mask_test   = np.array([s in test_subjects for s in subjects])
    mask_val    = np.array([s in val_subjects  for s in subjects])
    mask_train  = ~mask_test & ~mask_val

    if keep_classes is not None:
        keep = np.isin(y, keep_classes)
        mask_train &= keep
        mask_val   &= keep
        mask_test  &= keep

    return all_idx[mask_train], all_idx[mask_val], all_idx[mask_test]


def make_kfold_splits(
    h5_path: str | Path,
    k: int = 5,
    val_subjects: int = 3,
    seed: int = 42,
    keep_classes=None,
):
    """
    K-fold cross-validation POR SUJETO — el protocolo del paper UTA-RLDD.

    Particiona los sujetos en k grupos. Cada grupo es test exactamente una vez.
    Del resto de cada fold se reservan `val_subjects` sujetos para selección de
    checkpoint (sin fuga: val y test nunca comparten sujeto con train).

    Yields:
        (fold_idx, train_idx, val_idx, test_idx) por cada uno de los k folds.
    """
    with h5py.File(h5_path, "r") as f:
        y        = f["y"][:]
        subjects = f["subjects"][:].astype(str)

    rng = np.random.default_rng(seed)
    unique_subjects = np.unique(subjects)
    rng.shuffle(unique_subjects)

    folds = np.array_split(unique_subjects, k)
    all_idx = np.arange(len(y))
    keep = np.isin(y, keep_classes) if keep_classes is not None else np.ones(len(y), bool)

    for i in range(k):
        test_subj = set(folds[i].tolist())
        rest = np.array([s for s in unique_subjects if s not in test_subj])
        rng_fold = np.random.default_rng(seed + i)
        rng_fold.shuffle(rest)
        val_subj   = set(rest[:val_subjects].tolist())
        train_subj = set(rest[val_subjects:].tolist())

        mask_test  = np.array([s in test_subj  for s in subjects]) & keep
        mask_val   = np.array([s in val_subj   for s in subjects]) & keep
        mask_train = np.array([s in train_subj for s in subjects]) & keep

        yield i, all_idx[mask_train], all_idx[mask_val], all_idx[mask_test]


def video_level_eval(
    h5_path: str | Path,
    test_idx: np.ndarray,
    frame_preds: np.ndarray,
    label_map=None,
):
    """
    Agrega predicciones frame-a-frame a nivel de VIDEO (voto mayoritario) —
    cada video = (sujeto, clase original). Es la métrica que reporta el paper.

    Args:
        test_idx:    índices h5 evaluados (mismo orden que frame_preds)
        frame_preds: predicción por frame (ya en el espacio mapeado: 0/1)
        label_map:   para mapear la clase original del video al espacio de preds

    Retorna:
        dict con video_accuracy, video_f1, n_videos, y listas (true, pred) por video.
    """
    from sklearn.metrics import accuracy_score, f1_score

    with h5py.File(h5_path, "r") as f:
        y        = f["y"][test_idx].astype(int)
        subjects = f["subjects"][test_idx].astype(str)

    # video id = sujeto + clase original (cada sujeto tiene 1 video por clase)
    video_ids = np.array([f"{s}_{lbl}" for s, lbl in zip(subjects, y)])

    vid_true, vid_pred = [], []
    for vid in np.unique(video_ids):
        mask = video_ids == vid
        orig_label = int(y[mask][0])
        true = label_map[orig_label] if label_map is not None else orig_label
        pred = int(np.bincount(frame_preds[mask]).argmax())   # voto mayoritario
        vid_true.append(true)
        vid_pred.append(pred)

    vid_true, vid_pred = np.array(vid_true), np.array(vid_pred)
    return {
        "video_accuracy": float(accuracy_score(vid_true, vid_pred)),
        "video_f1":       float(f1_score(vid_true, vid_pred, average="macro", zero_division=0)),
        "n_videos":       len(vid_true),
        "true":           vid_true.tolist(),
        "pred":           vid_pred.tolist(),
    }


def make_loaders(
    h5_path: str | Path,
    train_idx: np.ndarray,
    val_idx:   np.ndarray,
    test_idx:  np.ndarray,
    batch_size: int = 32,
    num_workers: int = 4,
    label_map=None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Construye DataLoaders con augmentation en train y normalización en todos."""
    train_ds = DrowsinessDataset(h5_path, train_idx, transform=build_transforms(augment=True), label_map=label_map)
    val_ds   = DrowsinessDataset(h5_path, val_idx,   transform=build_transforms(augment=False), label_map=label_map)
    test_ds  = DrowsinessDataset(h5_path, test_idx,  transform=build_transforms(augment=False), label_map=label_map)

    persistent = num_workers > 0
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True, persistent_workers=persistent, prefetch_factor=2 if persistent else None)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=persistent, prefetch_factor=2 if persistent else None)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=persistent, prefetch_factor=2 if persistent else None)

    return train_loader, val_loader, test_loader
