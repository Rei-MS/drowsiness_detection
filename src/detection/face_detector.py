import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
)
_MODEL_PATH = Path(__file__).parent / "blaze_face_short_range.tflite"


def _ensure_model() -> Path:
    if not _MODEL_PATH.exists():
        print(f"Descargando modelo FaceDetector → {_MODEL_PATH} ...")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        print("Descarga completa.")
    return _MODEL_PATH


@dataclass
class BoundingBox:
    xmin: float
    ymin: float
    width: float
    height: float
    confidence: float

    def to_pixels(self, frame_w: int, frame_h: int) -> tuple[int, int, int, int]:
        """Retorna (x, y, w, h) en píxeles. Acepta coordenadas normalizadas o absolutas."""
        # Tasks API devuelve píxeles absolutos — normalizar si necesario
        x = int(self.xmin) if self.xmin > 1.0 else int(self.xmin * frame_w)
        y = int(self.ymin) if self.ymin > 1.0 else int(self.ymin * frame_h)
        w = int(self.width) if self.width > 1.0 else int(self.width * frame_w)
        h = int(self.height) if self.height > 1.0 else int(self.height * frame_h)
        return x, y, w, h


class FaceDetector:
    """Wrapper de MediaPipe FaceDetector (Tasks API ≥0.10)."""

    def __init__(self, model_selection: int = 0, min_confidence: float = 0.5):
        model_path = _ensure_model()
        base_options = mp_tasks.BaseOptions(
            model_asset_path=str(model_path),
            delegate=mp_tasks.BaseOptions.Delegate.CPU,
        )
        options = mp_vision.FaceDetectorOptions(
            base_options=base_options,
            min_detection_confidence=min_confidence,
            running_mode=mp_vision.RunningMode.IMAGE,
        )
        self._detector = mp_vision.FaceDetector.create_from_options(options)

    def detect(self, frame_bgr: np.ndarray) -> Optional[BoundingBox]:
        """Detecta el rostro más prominente en el frame. Retorna None si no hay detección."""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self._detector.detect(mp_image)
        if not result.detections:
            return None
        det = result.detections[0]
        bb = det.bounding_box
        score = det.categories[0].score if det.categories else 0.0
        return BoundingBox(
            xmin=float(bb.origin_x),
            ymin=float(bb.origin_y),
            width=float(bb.width),
            height=float(bb.height),
            confidence=float(score),
        )

    def draw(self, frame_bgr: np.ndarray, bbox: BoundingBox) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        x, y, bw, bh = bbox.to_pixels(w, h)
        cv2.rectangle(frame_bgr, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        cv2.putText(
            frame_bgr,
            f"{bbox.confidence:.2f}",
            (x, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
        return frame_bgr

    def close(self):
        self._detector.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
