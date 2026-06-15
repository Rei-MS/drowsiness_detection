"""
Entrenamiento del modelo temporal desde terminal.

Uso:
    source /home/lilidl/pnl/.venv/bin/activate
    cd /home/lilidl/tp_final_cv2
    python train_temporal.py
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.dataset import make_splits
from src.data.sequence_dataset import make_sequence_loaders
from src.models.temporal_model import TemporalDrowsinessModel
from src.training.train_temporal import train_temporal

H5            = "/home/lilidl/drowsiness_crops.h5"
BACKBONE_CKPT = "checkpoints/resnet50v2_best.pt"
KEEP_CLASSES  = [0, 2]
LABEL_MAP     = {0: 0, 2: 1}
SEQ_LEN       = 16     # ~6.4 s a 2.5 fps
TRAIN_STRIDE  = 4      # overlapping → más muestras
EVAL_STRIDE   = 16     # non-overlapping
BATCH_SIZE    = 32
NUM_WORKERS   = 8

Path("checkpoints").mkdir(exist_ok=True)
Path("docs").mkdir(exist_ok=True)

# ── Splits ────────────────────────────────────────────────────────────────────
print("Cargando splits...")
train_idx, val_idx, test_idx = make_splits(H5, keep_classes=KEEP_CLASSES)
np.save("data/processed/test_idx.npy", test_idx)

print("Construyendo sequence loaders...")
train_loader, val_loader, test_loader = make_sequence_loaders(
    H5, train_idx, val_idx, test_idx,
    seq_len=SEQ_LEN, train_stride=TRAIN_STRIDE, eval_stride=EVAL_STRIDE,
    batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, label_map=LABEL_MAP,
)
print(f"  train secuencias: {len(train_loader.dataset):,}")
print(f"  val   secuencias: {len(val_loader.dataset):,}")
print(f"  test  secuencias: {len(test_loader.dataset):,}")

# ── Modelo ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ENTRENANDO TemporalModel (ResNet50V2 + BiGRU + Attention)")
print("=" * 60)

model = TemporalDrowsinessModel(
    backbone_checkpoint=BACKBONE_CKPT,
    seq_len=SEQ_LEN,
    hidden=512,
    layers=2,
    dropout=0.3,
    num_classes=2,
)
print(f"Params totales:     {model.count_total():,}")
print(f"Entrenables fase 1: {model.count_trainable():,}")

history = train_temporal(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    checkpoint_path=Path("checkpoints/temporal_best.pt"),
    epochs_frozen=10,
    epochs_unfrozen=25,
    lr_head=1e-3,
    lr_finetune=5e-5,
    patience=8,
    unfreeze_blocks=2,
)

with open("checkpoints/history_temporal.json", "w") as f:
    json.dump(history, f, indent=2)

# ── Curvas ────────────────────────────────────────────────────────────────────
fig, (ax_loss, ax_f1) = plt.subplots(1, 2, figsize=(14, 5))
epochs = range(1, len(history["train_loss"]) + 1)
phase_change = next(
    i for i in range(1, len(history["phase"]))
    if history["phase"][i] != history["phase"][i - 1]
)

for ax, key, title in [(ax_loss, "loss", "Loss"), (ax_f1, "f1", "F1-macro")]:
    ax.plot(epochs, history[f"train_{key}"], "b--", alpha=0.6, label="train")
    ax.plot(epochs, history[f"val_{key}"],   "b-",              label="val")
    ax.axvline(phase_change + 0.5, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Epoch"); ax.set_ylabel(title)
    ax.set_title(f"Temporal — {title}"); ax.legend(); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("docs/curvas_temporal.png", dpi=150, bbox_inches="tight")
print("\nCurvas guardadas en docs/curvas_temporal.png")
print(f"Mejor val F1-macro: {max(history['val_f1']):.4f}")
