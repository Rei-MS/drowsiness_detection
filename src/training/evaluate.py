"""
Evaluación completa de un modelo sobre el test set.

Métricas calculadas:
  - F1-macro y F1 por clase
  - Matriz de confusión 3×3
  - Latencia CPU promedio (ms/frame)
  - Métricas temporales: consistencia, transiciones/minuto, dwell time
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from src.data.dataset import CLASS_NAMES
from src.models.backbone import DrowsinessModel
from src.models.temporal import TemporalSmoother


def load_checkpoint(checkpoint_path: Path, device: Optional[torch.device] = None) -> DrowsinessModel:
    """Carga un checkpoint y retorna el modelo en modo eval."""
    if device is None:
        device = torch.device("cpu")
    ckpt = torch.load(str(checkpoint_path), map_location=device)
    num_classes = ckpt.get("num_classes", len(CLASS_NAMES))
    model = DrowsinessModel(arch=ckpt["arch"], num_classes=num_classes, freeze=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model


def measure_latency(model: DrowsinessModel, input_size: tuple = (1, 3, 224, 224), n_runs: int = 100) -> float:
    """Latencia media de inferencia en CPU (ms/frame)."""
    cpu = torch.device("cpu")
    model_cpu = model.to(cpu).eval()
    dummy = torch.zeros(input_size)

    # warm-up
    for _ in range(5):
        with torch.no_grad():
            model_cpu(dummy)

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        with torch.no_grad():
            model_cpu(dummy)
        times.append((time.perf_counter() - t0) * 1000.0)

    return float(np.mean(times))


def temporal_metrics(
    preds: np.ndarray,
    fps: float = 30.0,
    num_classes: int = 3,
    window: int = 30,
    class_names: Optional[list] = None,
) -> dict:
    """
    Calcula métricas de estabilidad temporal sobre una secuencia de predicciones.

    Args:
        preds: array 1D de predicciones (una por frame)
        fps: frames por segundo del video original
        window: tamaño ventana para calcular consistencia

    Retorna:
        dict con: consistency, transitions_per_min, dwell_time_per_class
    """
    if class_names is None:
        class_names = CLASS_NAMES
    smoother = TemporalSmoother(window=window, num_classes=num_classes)
    smoothed = np.array([smoother.update(int(p)) for p in preds])

    # Consistencia: % de frames cuya pred coincide con la pred suavizada
    consistency = float((preds == smoothed).mean())

    # Transiciones: cambios de clase en la secuencia suavizada
    transitions = int(np.sum(smoothed[1:] != smoothed[:-1]))
    duration_min = len(preds) / fps / 60.0
    transitions_per_min = transitions / max(duration_min, 1e-6)

    # Dwell time: duración media (en segundos) de cada clase
    dwell: dict[int, list[float]] = {c: [] for c in range(num_classes)}
    i = 0
    while i < len(smoothed):
        cls = smoothed[i]
        j = i + 1
        while j < len(smoothed) and smoothed[j] == cls:
            j += 1
        dwell[cls].append((j - i) / fps)
        i = j

    dwell_time = {class_names[c]: float(np.mean(v)) if v else 0.0 for c, v in dwell.items()}

    return {
        "consistency": consistency,
        "transitions_per_min": transitions_per_min,
        "dwell_time_per_class": dwell_time,
    }


def evaluate(
    model: DrowsinessModel,
    test_loader: DataLoader,
    device: Optional[torch.device] = None,
    fps: float = 30.0,
    class_names: Optional[list] = None,
) -> dict:
    """
    Evaluación completa sobre el test set.

    Args:
        class_names: nombres de clase. Si None, usa CLASS_NAMES (3 clases).
            Para binario pasar p.ej. ['alerta', 'somnoliento'].

    Retorna:
        dict con todas las métricas.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if class_names is None:
        class_names = CLASS_NAMES
    n_classes = len(class_names)
    model.to(device).eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.numpy())

    all_preds  = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(n_classes)))
    report = classification_report(
        all_labels, all_preds,
        labels=list(range(n_classes)),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    latency_ms = measure_latency(model)
    temp = temporal_metrics(all_preds, fps=fps, num_classes=n_classes, class_names=class_names)

    results = {
        "f1_macro": f1_macro,
        "confusion_matrix": cm.tolist(),
        "per_class": {
            name: {
                "precision": report[name]["precision"],
                "recall":    report[name]["recall"],
                "f1":        report[name]["f1-score"],
                "support":   int(report[name]["support"]),
            }
            for name in class_names
        },
        "latency_cpu_ms": latency_ms,
        **temp,
    }

    _print_results(results)
    return results


def _print_results(r: dict) -> None:
    print(f"\n{'='*50}")
    print(f"F1-macro: {r['f1_macro']:.4f}")
    print(f"Latencia CPU: {r['latency_cpu_ms']:.1f} ms/frame ({1000/r['latency_cpu_ms']:.1f} fps)")
    print(f"\nPor clase:")
    for name, m in r["per_class"].items():
        print(f"  {name:20s}  P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}  n={m['support']}")
    print(f"\nConsistencia temporal: {r['consistency']:.3f}")
    print(f"Transiciones/min: {r['transitions_per_min']:.2f}")
    print(f"Dwell time: {r['dwell_time_per_class']}")
    print("="*50)
