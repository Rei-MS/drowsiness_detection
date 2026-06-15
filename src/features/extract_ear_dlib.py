"""
Extrae EAR real por frame con dlib (68 landmarks) para todos los crops del h5.

Los crops YA son caras (224x224), así que se le pasa el crop entero al shape
predictor sin re-detectar (más rápido y rescata frames donde el detector falla).
Como fallback, si el EAR sale degenerado se intenta una detección explícita.

Guarda un array `ear_dlib` (N,) en el MISMO orden que X/y/subjects, en un .npy
aparte (no toca el h5). EAR = NaN si no se pudo estimar.

Uso:
    python -m src.features.extract_ear_dlib \
        --h5 /home/lilidl/drowsiness_crops.h5 \
        --predictor models/shape_predictor_68_face_landmarks.dat \
        --output data/processed/ear_dlib.npy
"""

import argparse
import time
from pathlib import Path

import dlib
import h5py
import numpy as np

# Esquema 68-landmarks de dlib
L_EYE = [42, 43, 44, 45, 46, 47]
R_EYE = [36, 37, 38, 39, 40, 41]


def _ear(pts: np.ndarray) -> float:
    a = np.linalg.norm(pts[1] - pts[5])
    b = np.linalg.norm(pts[2] - pts[4])
    c = np.linalg.norm(pts[0] - pts[3])
    return float((a + b) / (2.0 * c + 1e-6))


def extract(h5_path: str, predictor_path: str, output_path: str, batch: int = 512) -> None:
    det = dlib.get_frontal_face_detector()
    sp  = dlib.shape_predictor(predictor_path)

    with h5py.File(h5_path, "r") as f:
        X = f["X"]
        n, h, w = X.shape[0], X.shape[1], X.shape[2]
        ear = np.full(n, np.nan, dtype=np.float32)
        full_rect = dlib.rectangle(0, 0, w, h)

        t0 = time.time()
        n_fallback = 0
        for start in range(0, n, batch):
            end = min(start + batch, n)
            block = X[start:end]   # lectura secuencial (rápida)
            for j in range(end - start):
                im = np.ascontiguousarray(block[j])
                # 1) crop entero como cara (rápido)
                shp = sp(im, full_rect)
                pts = np.array([[shp.part(k).x, shp.part(k).y] for k in range(68)], float)
                val = (_ear(pts[L_EYE]) + _ear(pts[R_EYE])) / 2.0

                # 2) fallback: detección explícita si el valor es degenerado
                if not (0.05 < val < 0.6):
                    faces = det(im, 0)
                    if faces:
                        shp = sp(im, faces[0])
                        pts = np.array([[shp.part(k).x, shp.part(k).y] for k in range(68)], float)
                        val = (_ear(pts[L_EYE]) + _ear(pts[R_EYE])) / 2.0
                        n_fallback += 1
                    else:
                        val = np.nan
                ear[start + j] = val

            done = end
            rate = done / max(time.time() - t0, 1e-6)
            eta = (n - done) / max(rate, 1e-6)
            print(f"\r  {done}/{n}  ({rate:.0f} crops/s, ETA {eta/60:.1f} min, "
                  f"fallback={n_fallback})", end="", flush=True)
        print()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, ear)

    valid = np.isfinite(ear)
    print(f"\nGuardado: {out}")
    print(f"  válidos: {valid.sum():,}/{n:,} ({100*valid.mean():.1f}%)")
    print(f"  EAR media (válidos): {np.nanmean(ear):.3f} ± {np.nanstd(ear):.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--h5", default="/home/lilidl/drowsiness_crops.h5")
    p.add_argument("--predictor", default="models/shape_predictor_68_face_landmarks.dat")
    p.add_argument("--output", default="data/processed/ear_dlib.npy")
    p.add_argument("--batch", type=int, default=512)
    a = p.parse_args()
    extract(a.h5, a.predictor, a.output, a.batch)
