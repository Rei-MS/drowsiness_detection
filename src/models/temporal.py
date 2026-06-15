"""
Suavizado temporal por ventana deslizante.

Reduce el ruido de predicciones frame a frame calculando la moda sobre
una ventana de N frames. Patrón de deque de TP/src/demo/realtime.py.
"""

from collections import deque

import numpy as np


class TemporalSmoother:
    """
    Ventana deslizante de N frames → moda como predicción suavizada.

    Args:
        window: tamaño de la ventana (default=30, ≈1 segundo a 30fps)
        num_classes: número de clases posibles
    """

    def __init__(self, window: int = 30, num_classes: int = 3):
        self._buf: deque[int] = deque(maxlen=window)
        self.num_classes = num_classes

    def update(self, pred: int) -> int:
        """
        Añade la predicción del frame actual y retorna la predicción suavizada.

        Returns:
            Clase más frecuente en la ventana actual.
        """
        self._buf.append(int(pred))
        counts = np.bincount(list(self._buf), minlength=self.num_classes)
        return int(counts.argmax())

    def probabilities(self) -> np.ndarray:
        """Distribución de frecuencias normalizada de las predicciones en la ventana."""
        if not self._buf:
            return np.ones(self.num_classes) / self.num_classes
        counts = np.bincount(list(self._buf), minlength=self.num_classes)
        return counts / counts.sum()

    def reset(self) -> None:
        self._buf.clear()

    @property
    def filled(self) -> bool:
        """True cuando la ventana ya tiene el máximo de frames."""
        return len(self._buf) == self._buf.maxlen
