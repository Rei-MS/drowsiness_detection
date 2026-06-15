"""
Temporal v2 — menos overfitting:
  - train_stride 4→8  (overlap 75%→50%)
  - dropout 0.3→0.5
  - lr_finetune 5e-5→1e-5
  - weight_decay 1e-4→5e-4
  - frame_drop en el modelo (zeroa features enteros)
  - frozen 10→15 epochs

Uso:
    source /home/lilidl/pnl/.venv/bin/activate
    cd /home/lilidl/tp_final_cv2
    python train_temporal_v2.py
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
SEQ_LEN       = 16
TRAIN_STRIDE  = 8     # era 4 → menos overlap
EVAL_STRIDE   = 16
BATCH_SIZE    = 32
NUM_WORKERS   = 8

Path("checkpoints").mkdir(exist_ok=True)
Path("docs").mkdir(exist_ok=True)

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

print("\n" + "=" * 60)
print("ENTRENANDO TemporalModel v2 (ResNet50V2 + BiGRU + Attention)")
print("=" * 60)

model = TemporalDrowsinessModel(
    backbone_checkpoint=BACKBONE_CKPT,
    seq_len=SEQ_LEN,
    hidden=512,
    layers=2,
    dropout=0.5,       # era 0.3
    num_classes=2,
)
print(f"Params totales:     {model.count_total():,}")
print(f"Entrenables fase 1: {model.count_trainable():,}")

history = train_temporal(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    checkpoint_path=Path("checkpoints/temporal_v2_best.pt"),
    epochs_frozen=15,          # era 10
    epochs_unfrozen=25,
    lr_head=1e-3,
    lr_finetune=1e-5,          # era 5e-5
    patience=10,               # más paciencia con lr más bajo
    unfreeze_blocks=2,
)

# Guardar history con weight_decay en metadata
history["config"] = {
    "train_stride": TRAIN_STRIDE,
    "dropout": 0.5,
    "lr_finetune": 1e-5,
    "weight_decay": 5e-4,
    "epochs_frozen": 15,
}

with open("checkpoints/history_temporal_v2.json", "w") as f:
    json.dump(history, f, indent=2)

# Comparación v1 vs v2
fig, (ax_loss, ax_f1) = plt.subplots(1, 2, figsize=(14, 5))

with open("checkpoints/history_temporal.json") as f:
    h_v1 = json.load(f)

for h, label, color in [(h_v1, "v1 (stride=4)", "b"), (history, "v2 (stride=8)", "r")]:
    epochs = range(1, len(h["train_loss"]) + 1)
    phase_change = next(
        i for i in range(1, len(h["phase"])) if h["phase"][i] != h["phase"][i - 1]
    )
    ax_loss.plot(epochs, h["val_loss"],  f"{color}-",  label=f"{label} val")
    ax_f1.plot(  epochs, h["val_f1"],    f"{color}-",  label=f"{label} val")
    ax_f1.plot(  epochs, h["train_f1"],  f"{color}--", alpha=0.4, label=f"{label} train")
    for ax in (ax_loss, ax_f1):
        ax.axvline(phase_change + 0.5, color=color, linestyle=":", alpha=0.3)

ax_loss.set_xlabel("Epoch"); ax_loss.set_ylabel("Loss"); ax_loss.set_title("Val Loss — v1 vs v2")
ax_loss.legend(); ax_loss.grid(alpha=0.3)
ax_f1.set_xlabel("Epoch"); ax_f1.set_ylabel("F1-macro"); ax_f1.set_title("F1-macro — v1 vs v2")
ax_f1.legend(); ax_f1.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("docs/curvas_temporal_v2.png", dpi=150, bbox_inches="tight")
print("\nCurvas guardadas en docs/curvas_temporal_v2.png")
print(f"v1 mejor val F1: {max(h_v1['val_f1']):.4f}")
print(f"v2 mejor val F1: {max(history['val_f1']):.4f}")
