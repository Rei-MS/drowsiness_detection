"""
Eye Aspect Ratio (EAR) — Soukupová & Čech (2016).

EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

Landmarks MediaPipe FaceMesh por ojo:
    [p1, p2, p3, p4, p5, p6] en sentido horario desde el canto externo.
"""

from collections import deque

import numpy as np

# Índices MediaPipe FaceMesh para cada ojo (6 puntos en orden: externo, sup-ext,
# sup-int, interno, inf-int, inf-ext)
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

EAR_CLOSED_THRESHOLD = 0.20  # EAR por debajo → ojo cerrado


def _eye_aspect_ratio(landmarks: np.ndarray, eye_indices: list[int]) -> float:
    pts = landmarks[eye_indices, :2]  # solo x, y
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    return float((A + B) / (2.0 * C + 1e-6))


def compute_ear(landmarks: np.ndarray) -> dict[str, float]:
    """
    Calcula EAR para ojo izquierdo, derecho y promedio.

    Args:
        landmarks: array (468, 3) de MediaPipe FaceMesh.

    Returns:
        dict con claves 'left', 'right', 'avg'.
    """
    left = _eye_aspect_ratio(landmarks, LEFT_EYE)
    right = _eye_aspect_ratio(landmarks, RIGHT_EYE)
    return {"left": left, "right": right, "avg": (left + right) / 2.0}


class BlinkDetector:
    """Detecta parpadeos y calcula tasa de parpadeo (blinks/minuto)."""

    def __init__(self, threshold: float = EAR_CLOSED_THRESHOLD, consec_frames: int = 2):
        self.threshold = threshold
        self.consec_frames = consec_frames
        self._counter = 0       # frames consecutivos con ojo cerrado
        self._total = 0         # parpadeos totales
        self._history: deque[float] = deque(maxlen=1800)  # ~1 min a 30fps

    def update(self, ear_avg: float) -> bool:
        """
        Actualiza el estado con el EAR promedio del frame actual.

        Returns:
            True si se detectó un parpadeo en este frame.
        """
        self._history.append(ear_avg)
        blink = False
        if ear_avg < self.threshold:
            self._counter += 1
        else:
            if self._counter >= self.consec_frames:
                self._total += 1
                blink = True
            self._counter = 0
        return blink

    @property
    def total_blinks(self) -> int:
        return self._total

    def blink_rate(self, fps: float = 30.0) -> float:
        """Parpadeos por minuto estimados sobre el historial disponible."""
        if len(self._history) < 2:
            return 0.0
        elapsed_seconds = len(self._history) / fps
        return (self._total / elapsed_seconds) * 60.0
