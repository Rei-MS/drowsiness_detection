"""
Extracción de crops faciales del dataset UTA-RLDD.

Estructura esperada en data/raw/ (organizada por sujeto):
    01/  →  0.mov   "alerta"          (label 0)
            5.mov   "baja vigilancia" (label 1)
            10.MOV  "somnoliento"     (label 2)
    02/  →  idem
    ...
    60/  →  idem

Las extensiones pueden ser .mov, .MOV, .mp4 o .avi.

Salida: data/processed/crops.h5 (HDF5, escritura incremental — no acumula en RAM).
    X        (N, 224, 224, 3)  uint8  crops RGB
    y        (N,)              int64  0/1/2
    subjects (N,)              str    ID sujeto (nombre de carpeta)
    ear      (N,)              float32 EAR estimado
"""

import argparse
from pathlib import Path

import cv2
import h5py
import numpy as np

from src.detection.face_mesh import FaceMesh
from src.features.ear import compute_ear

LABEL_MAP = {"0": 0, "5": 1, "10": 2}
CROP_SIZE = 224
FRAME_STRIDE = 12
FACE_MARGIN = 0.25
CHUNK = 512   # frames por bloque HDF5


def _find_video(subject_dir: Path, level: str) -> Path | None:
    for f in subject_dir.iterdir():
        if (f.stem == level or f.stem.startswith(level + "_")) and f.suffix.lower() in (".mov", ".avi", ".mp4", ".m4v"):
            return f
    return None


def _landmarks_to_bbox(landmarks: np.ndarray, frame_h: int, frame_w: int) -> tuple[int, int, int, int]:
    px = landmarks[:, :2].copy()
    px[:, 0] *= frame_w
    px[:, 1] *= frame_h
    x_min, y_min = px.min(axis=0).astype(int)
    x_max, y_max = px.max(axis=0).astype(int)
    margin_x = int((x_max - x_min) * FACE_MARGIN)
    margin_y = int((y_max - y_min) * FACE_MARGIN)
    return (
        max(0, x_min - margin_x),
        max(0, y_min - margin_y),
        min(frame_w, x_max + margin_x),
        min(frame_h, y_max + margin_y),
    )


def _append(ds: h5py.Dataset, data: np.ndarray) -> None:
    """Extiende un dataset HDF5 resizable en el eje 0."""
    n = len(data)
    cur = ds.shape[0]
    ds.resize(cur + n, axis=0)
    ds[cur:] = data


def process_video(
    video_path: Path,
    label: int,
    subject_id: str,
    mesh: FaceMesh,
    h5_datasets: tuple,
    stride: int = FRAME_STRIDE,
) -> int:
    """Extrae crops de un video y los escribe directamente en HDF5. Retorna nº de frames."""
    ds_X, ds_y, ds_subj, ds_ear = h5_datasets
    buf_X, buf_y, buf_subj, buf_ear = [], [], [], []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [WARN] No se pudo abrir: {video_path.name}")
        return 0

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % stride != 0:
            frame_idx += 1
            continue

        h, w = frame.shape[:2]
        landmarks = mesh.process(frame)
        if landmarks is None:
            frame_idx += 1
            continue

        x1, y1, x2, y2 = _landmarks_to_bbox(landmarks, h, w)
        if x2 <= x1 or y2 <= y1:
            frame_idx += 1
            continue

        crop = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
        crop = cv2.resize(crop, (CROP_SIZE, CROP_SIZE))

        ear_avg = float(compute_ear(landmarks)["avg"])

        buf_X.append(crop)
        buf_y.append(label)
        buf_subj.append(subject_id)
        buf_ear.append(ear_avg)

        # Volcar al disco por bloques para no acumular en RAM
        if len(buf_X) >= CHUNK:
            _append(ds_X,    np.array(buf_X, dtype=np.uint8))
            _append(ds_y,    np.array(buf_y, dtype=np.int64))
            _append(ds_subj, np.array(buf_subj, dtype=object))
            _append(ds_ear,  np.array(buf_ear, dtype=np.float32))
            buf_X, buf_y, buf_subj, buf_ear = [], [], [], []

        frame_idx += 1

    cap.release()

    # Volcar resto
    if buf_X:
        _append(ds_X,    np.array(buf_X, dtype=np.uint8))
        _append(ds_y,    np.array(buf_y, dtype=np.int64))
        _append(ds_subj, np.array(buf_subj, dtype=object))
        _append(ds_ear,  np.array(buf_ear, dtype=np.float32))

    return frame_idx  # frames procesados (no todos son crops válidos)


def extract_all(data_dir: Path, output_path: Path, stride: int = FRAME_STRIDE) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    subject_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    print(f"Sujetos encontrados: {len(subject_dirs)}")

    with h5py.File(output_path, "w") as f:
        ds_X = f.create_dataset(
            "X", shape=(0, CROP_SIZE, CROP_SIZE, 3),
            maxshape=(None, CROP_SIZE, CROP_SIZE, 3),
            dtype="uint8", chunks=(CHUNK, CROP_SIZE, CROP_SIZE, 3),
        )
        ds_y    = f.create_dataset("y",        shape=(0,), maxshape=(None,), dtype="int64")
        ds_subj = f.create_dataset("subjects", shape=(0,), maxshape=(None,), dtype=h5py.string_dtype())
        ds_ear  = f.create_dataset("ear",      shape=(0,), maxshape=(None,), dtype="float32")
        dsets = (ds_X, ds_y, ds_subj, ds_ear)

        with FaceMesh() as mesh:
            for subject_dir in subject_dirs:
                subject_id = subject_dir.name
                print(f"\nSujeto {subject_id}:")

                for level_name, label in LABEL_MAP.items():
                    video_path = _find_video(subject_dir, level_name)
                    if video_path is None:
                        print(f"  [WARN] nivel {level_name} no encontrado")
                        continue

                    n_before = ds_X.shape[0]
                    print(f"  nivel {level_name} → {video_path.name} ...", end=" ", flush=True)
                    process_video(video_path, label, subject_id, mesh, dsets, stride)
                    n_crops = ds_X.shape[0] - n_before
                    print(f"{n_crops} crops")

        total = ds_X.shape[0]
        y_all = ds_y[:]
        subj_all = ds_subj[:].astype(str)

    print(f"\nGuardado en {output_path}")
    print(f"  Total crops: {total}")
    for lbl, name in [(0, "alerta"), (1, "baja_vigilancia"), (2, "somnoliento")]:
        print(f"  clase {lbl} ({name}): {(y_all == lbl).sum()}")
    print(f"  Sujetos únicos: {len(np.unique(subj_all))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--output",   default="data/processed/crops.h5")
    parser.add_argument("--stride",   type=int, default=FRAME_STRIDE)
    args = parser.parse_args()

    extract_all(Path(args.data_dir), Path(args.output), stride=args.stride)
