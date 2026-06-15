"""
Reemplazo sin mediapipe usando Haar cascade de OpenCV.

Retorna un array de 478 landmarks sintéticos compatible con:
  - _landmarks_to_bbox (usa todos los puntos para el bbox de la cara)
  - compute_ear (índices 33,160,158,133,153,144 y 362,385,387,263,373,380
    estimados desde proporciones faciales típicas)

Los valores EAR son aproximados; no se usan en el entrenamiento (dataset.py
solo consume X e y), pero se almacenan en crops.h5 para análisis.
"""

import cv2
import numpy as np
from typing import Optional

# Proporciones faciales para estimar posición de ojos
_EYE_Y_RATIO = 0.38   # ojos al 38% del alto de la cara desde arriba
_EYE_SPREAD_Y = 0.04  # spread vertical de los 6 puntos EAR (fracción del alto)
_EYE_HALF_W = 0.07    # semiancho del ojo (fracción del ancho de la cara)
_RIGHT_EYE_X = 0.30   # centro ojo derecho (fracción del ancho desde borde izq)
_LEFT_EYE_X = 0.70    # centro ojo izquierdo


class FaceMesh:
    """Detector de cara + landmarks sintéticos. Misma interfaz que la versión mediapipe."""

    def __init__(
        self,
        max_faces: int = 1,
        refine_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._detector = cv2.CascadeClassifier(cascade_path)

    def process(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Detecta la cara más grande y retorna array (478, 3) de landmarks normalizados.
        Retorna None si no detecta ninguna cara.
        """
        h, w = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = self._detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )
        if len(faces) == 0:
            return None

        fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])

        # Grid de puntos sobre el bounding box (cubre el bbox para _landmarks_to_bbox)
        lms = np.zeros((478, 3), dtype=np.float32)
        xs = np.linspace(fx / w, (fx + fw) / w, 22)
        ys = np.linspace(fy / h, (fy + fh) / h, 22)
        idx = 0
        for xi in xs:
            for yi in ys:
                if idx < 478:
                    lms[idx] = [xi, yi, 0.0]
                    idx += 1

        # Landmarks de ojo para EAR (estimados desde proporciones)
        x0, y0 = fx / w, fy / h
        fw_n, fh_n = fw / w, fh / h

        ey = y0 + fh_n * _EYE_Y_RATIO
        ey_s = fh_n * _EYE_SPREAD_Y
        ex_h = fw_n * _EYE_HALF_W

        for indices, cx in (
            ([33, 160, 158, 133, 153, 144], x0 + fw_n * _RIGHT_EYE_X),
            ([362, 385, 387, 263, 373, 380], x0 + fw_n * _LEFT_EYE_X),
        ):
            pts = [
                [cx - ex_h,         ey,         0],
                [cx - ex_h * 0.5,   ey - ey_s,  0],
                [cx + ex_h * 0.5,   ey - ey_s,  0],
                [cx + ex_h,         ey,         0],
                [cx + ex_h * 0.5,   ey + ey_s,  0],
                [cx - ex_h * 0.5,   ey + ey_s,  0],
            ]
            for i, p in zip(indices, pts):
                lms[i] = p

        return lms

    def draw(self, frame_bgr: np.ndarray) -> np.ndarray:
        return frame_bgr

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def get_landmark_coords(landmarks: np.ndarray, indices: list[int]) -> np.ndarray:
    return landmarks[indices]


def to_pixels(landmarks: np.ndarray, frame_w: int, frame_h: int) -> np.ndarray:
    px = landmarks[:, :2].copy()
    px[:, 0] *= frame_w
    px[:, 1] *= frame_h
    return px.astype(np.int32)
