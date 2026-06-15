"""
Paso 2 — EAR denso desde los videos crudos a ~15 fps (stride=2), en paralelo.

Para cada video (sujeto, nivel) extrae la secuencia de EAR real con
Haar (box, downscale 480px) + dlib 68-landmarks. NO guarda crops, solo el
EAR escalar por frame → permite features de parpadeo reales (rate, duración,
PERCLOS fino) que a stride=12 estaban submuestreadas.

Salida: data/processed/ear_dense.npz con arrays alineados:
    ear, subject, label, frame_idx, video_id

Uso:
    python -m src.features.extract_ear_dense --workers 6 --stride 2
"""

import argparse
import time
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import cv2
import dlib
import numpy as np

from src.data.extract_crops import LABEL_MAP, _find_video

PREDICTOR = "models/shape_predictor_68_face_landmarks.dat"
L_EYE = [42, 43, 44, 45, 46, 47]
R_EYE = [36, 37, 38, 39, 40, 41]
DW = 480          # ancho de downscale para detección+landmarks (EAR es invariante a escala)
REDETECT = 10     # re-detectar la cara cada N frames procesados (resto: reusar box)

# globals por worker (se inicializan una vez por proceso)
_haar = None
_sp = None


def _init_worker():
    global _haar, _sp
    _haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    _sp = dlib.shape_predictor(PREDICTOR)


def _ear(p: np.ndarray) -> float:
    a = np.linalg.norm(p[1] - p[5])
    b = np.linalg.norm(p[2] - p[4])
    c = np.linalg.norm(p[0] - p[3])
    return float((a + b) / (2.0 * c + 1e-6))


def _process_video(task, stride: int):
    subject, level, video_path = task
    label = LABEL_MAP[level]
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return subject, label, np.array([]), np.array([])

    ears, idxs = [], []
    last_box = None
    idx = 0
    proc = 0
    while True:
        ret, fr = cap.read()
        if not ret:
            break
        if idx % stride != 0:
            idx += 1
            continue
        h, w = fr.shape[:2]
        scale = DW / w
        small = cv2.resize(fr, (DW, int(h * scale)))
        gray = cv2.equalizeHist(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))

        if last_box is None or proc % REDETECT == 0:
            faces = _haar.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
            if len(faces):
                last_box = max(faces, key=lambda f: f[2] * f[3])

        if last_box is not None:
            x, y, fw, fh = last_box
            shp = _sp(small, dlib.rectangle(int(x), int(y), int(x + fw), int(y + fh)))
            pts = np.array([[shp.part(k).x, shp.part(k).y] for k in range(68)], float)
            ears.append((_ear(pts[L_EYE]) + _ear(pts[R_EYE])) / 2.0)
            idxs.append(idx)
        idx += 1
        proc += 1

    cap.release()
    return subject, label, np.array(ears, np.float32), np.array(idxs, np.int32)


def main(workers: int, stride: int, output: str):
    raw = Path("data/raw")
    tasks = []
    for sd in sorted(d for d in raw.iterdir() if d.is_dir()):
        for level in LABEL_MAP:
            vp = _find_video(sd, level)
            if vp is not None:
                tasks.append((sd.name, level, vp))
    print(f"Videos a procesar: {len(tasks)}  (workers={workers}, stride={stride})")

    all_ear, all_subj, all_lbl, all_idx, all_vid = [], [], [], [], []
    t0 = time.time()
    with Pool(workers, initializer=_init_worker) as pool:
        fn = partial(_process_video, stride=stride)
        for n, (subject, label, ears, idxs) in enumerate(pool.imap_unordered(fn, tasks), 1):
            vid = f"{subject}_{label}"
            all_ear.append(ears)
            all_idx.append(idxs)
            all_subj.append(np.full(len(ears), subject, dtype=object))
            all_lbl.append(np.full(len(ears), label, dtype=np.int64))
            all_vid.append(np.full(len(ears), vid, dtype=object))
            print(f"\r  {n}/{len(tasks)} videos  ({time.time()-t0:.0f}s)  "
                  f"último: {vid} ({len(ears)} frames)", end="", flush=True)
    print()

    ear = np.concatenate(all_ear)
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        ear=ear,
        subject=np.concatenate(all_subj).astype(str),
        label=np.concatenate(all_lbl),
        frame_idx=np.concatenate(all_idx),
        video_id=np.concatenate(all_vid).astype(str),
    )
    print(f"\nGuardado: {out}  ({len(ear):,} frames con EAR válido)")
    print(f"  EAR media: {np.nanmean(ear):.3f} ± {np.nanstd(ear):.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--output", default="data/processed/ear_dense.npz")
    a = p.parse_args()
    main(a.workers, a.stride, a.output)
