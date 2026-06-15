"""
Demo de inferencia en tiempo real — comparación side-by-side MobileNetV2 vs ResNet50V2.

Uso:
    python -m src.inference.demo --model-a checkpoints/mobilenetv2_best.pt \
                                  --model-b checkpoints/resnet50v2_best.pt \
                                  --source 0

    --source 0          → webcam
    --source video.mp4  → archivo de video

Controles:
    q  → salir
    s  → capturar screenshot (screenshot_<N>.png)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from src.data.dataset import IMAGENET_MEAN, IMAGENET_STD
from src.detection.face_mesh import FaceMesh
from src.features.ear import compute_ear
from src.models.backbone import DrowsinessModel
from src.models.temporal import TemporalSmoother
from src.training.evaluate import load_checkpoint

# ── Configuración visual ─────────────────────────────────────────────────────
CLASS_NAMES  = ["ALERTA", "BAJA VIG.", "SOMNOLIENTO"]
CLASS_COLORS = [(0, 200, 50), (0, 180, 220), (0, 50, 220)]  # BGR: verde, amarillo, rojo

FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_SMALL = 0.55
FONT_MED   = 0.75
FONT_BOLD  = cv2.LINE_AA

PANEL_W = 420  # ancho de cada panel lateral (se escala si el frame es distinto)


# ── Preprocesamiento ─────────────────────────────────────────────────────────
_mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 3)
_std  = np.array(IMAGENET_STD,  dtype=np.float32).reshape(1, 1, 3)


def preprocess(crop_rgb: np.ndarray) -> torch.Tensor:
    """Convierte crop uint8 RGB a tensor normalizado (1, 3, 224, 224)."""
    img = crop_rgb.astype(np.float32) / 255.0
    img = (img - _mean) / _std
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)
    return tensor


# ── Extracción de crop ────────────────────────────────────────────────────────
def _crop_from_landmarks(landmarks: np.ndarray, frame_bgr: np.ndarray, margin: float = 0.25):
    h, w = frame_bgr.shape[:2]
    px = landmarks[:, :2].copy()
    px[:, 0] *= w
    px[:, 1] *= h
    x_min, y_min = px.min(axis=0).astype(int)
    x_max, y_max = px.max(axis=0).astype(int)
    mx = int((x_max - x_min) * margin)
    my = int((y_max - y_min) * margin)
    x1 = max(0, x_min - mx)
    y1 = max(0, y_min - my)
    x2 = min(w, x_max + mx)
    y2 = min(h, y_max + my)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame_bgr[y1:y2, x1:x2]
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    return cv2.resize(crop_rgb, (224, 224))


# ── Overlay helpers ───────────────────────────────────────────────────────────
def _draw_panel(
    canvas: np.ndarray,
    x_offset: int,
    model_name: str,
    pred_raw: int,
    pred_smooth: int,
    confidence: float,
    ear: float,
    fps: float,
) -> None:
    """Dibuja el panel de información de un modelo sobre canvas."""
    color = CLASS_COLORS[pred_smooth]

    # Barra de color en top
    cv2.rectangle(canvas, (x_offset, 0), (x_offset + PANEL_W, 6), color, -1)

    # Nombre del modelo
    cv2.putText(canvas, model_name, (x_offset + 10, 30),
                FONT, FONT_MED, (220, 220, 220), 2, FONT_BOLD)

    # Clase suavizada (grande)
    label = CLASS_NAMES[pred_smooth]
    cv2.putText(canvas, label, (x_offset + 10, 65),
                FONT, FONT_MED, color, 2, FONT_BOLD)

    # Confianza
    cv2.putText(canvas, f"Conf: {confidence:.2f}", (x_offset + 10, 95),
                FONT, FONT_SMALL, (200, 200, 200), 1, FONT_BOLD)

    # Raw (sin suavizar)
    cv2.putText(canvas, f"Raw: {CLASS_NAMES[pred_raw]}", (x_offset + 10, 118),
                FONT, FONT_SMALL, (150, 150, 150), 1, FONT_BOLD)

    # EAR (solo en panel izquierdo para no repetir)
    if x_offset < PANEL_W:
        ear_color = (0, 200, 50) if ear > 0.20 else (0, 50, 220)
        cv2.putText(canvas, f"EAR: {ear:.3f}", (x_offset + 10, 145),
                    FONT, FONT_SMALL, ear_color, 1, FONT_BOLD)

    # FPS
    cv2.putText(canvas, f"FPS: {fps:.1f}", (x_offset + 10, 168),
                FONT, FONT_SMALL, (160, 160, 160), 1, FONT_BOLD)


# ── Loop principal ────────────────────────────────────────────────────────────
def run_demo(
    model_a: DrowsinessModel,
    model_b: DrowsinessModel,
    source: str | int,
    window_size: int = 30,
) -> None:
    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la fuente: {source}")

    smoother_a = TemporalSmoother(window=window_size)
    smoother_b = TemporalSmoother(window=window_size)
    device = next(model_a.parameters()).device
    model_a.eval()
    model_b.eval()

    ema_fps = None
    screenshot_idx = 0

    with FaceMesh() as mesh:
        while True:
            t0 = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            landmarks = mesh.process(frame)

            if landmarks is not None:
                crop = _crop_from_landmarks(landmarks, frame)
                ear_vals = compute_ear(landmarks)
                ear = float(ear_vals["avg"])

                if crop is not None:
                    tensor = preprocess(crop).to(device)
                    with torch.no_grad():
                        logits_a = model_a(tensor)
                        logits_b = model_b(tensor)

                    probs_a = F.softmax(logits_a, dim=1)[0].cpu().numpy()
                    probs_b = F.softmax(logits_b, dim=1)[0].cpu().numpy()

                    raw_a = int(probs_a.argmax())
                    raw_b = int(probs_b.argmax())
                    conf_a = float(probs_a.max())
                    conf_b = float(probs_b.max())

                    smooth_a = smoother_a.update(raw_a)
                    smooth_b = smoother_b.update(raw_b)
                else:
                    raw_a = raw_b = smooth_a = smooth_b = 0
                    conf_a = conf_b = 0.0
                    ear = 0.0

                # Dibujar landmarks en frame
                frame = mesh.draw(frame)
            else:
                raw_a = raw_b = smooth_a = smooth_b = 0
                conf_a = conf_b = 0.0
                ear = 0.0

            # FPS (EMA)
            elapsed = time.perf_counter() - t0
            fps_inst = 1.0 / max(elapsed, 1e-6)
            ema_fps = fps_inst if ema_fps is None else 0.9 * ema_fps + 0.1 * fps_inst

            # Construir canvas side-by-side: [panel_A | frame | panel_B]
            target_h = 480
            scale = target_h / h if h > 0 else 1.0
            new_w = int(w * scale)
            frame_resized = cv2.resize(frame, (new_w, target_h))

            canvas = np.zeros((target_h, 2 * PANEL_W + new_w, 3), dtype=np.uint8)
            canvas[:, PANEL_W:PANEL_W + new_w] = frame_resized

            _draw_panel(canvas, 0,           "MobileNetV2",  raw_a, smooth_a, conf_a, ear, ema_fps)
            _draw_panel(canvas, PANEL_W + new_w, "ResNet50V2", raw_b, smooth_b, conf_b, ear, ema_fps)

            # Separadores verticales
            cv2.line(canvas, (PANEL_W, 0), (PANEL_W, target_h), (80, 80, 80), 1)
            cv2.line(canvas, (PANEL_W + new_w, 0), (PANEL_W + new_w, target_h), (80, 80, 80), 1)

            cv2.imshow("Drowsiness Detection — MobileNetV2 vs ResNet50V2", canvas)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                fname = f"screenshot_{screenshot_idx:03d}.png"
                cv2.imwrite(fname, canvas)
                print(f"Screenshot guardado: {fname}")
                screenshot_idx += 1

    cap.release()
    cv2.destroyAllWindows()


# ── Entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Demo side-by-side: MobileNetV2 vs ResNet50V2")
    parser.add_argument("--model-a", required=True, help="Checkpoint MobileNetV2")
    parser.add_argument("--model-b", required=True, help="Checkpoint ResNet50V2")
    parser.add_argument("--source", default="0", help="0=webcam o path a video")
    parser.add_argument("--window", type=int, default=30, help="Ventana temporal en frames")
    parser.add_argument("--device", default=None, help="cuda o cpu (default: auto)")
    args = parser.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Cargando modelos en {device}...")
    model_a = load_checkpoint(Path(args.model_a), device=device)
    model_b = load_checkpoint(Path(args.model_b), device=device)

    print("Iniciando demo. Presioná 'q' para salir, 's' para screenshot.")
    run_demo(model_a, model_b, source=args.source, window_size=args.window)


if __name__ == "__main__":
    main()
