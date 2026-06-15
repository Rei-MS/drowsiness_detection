"""
Equivalente al 02_training.ipynb — corre directo desde terminal.

Uso:
    source /home/lilidl/pnl/.venv/bin/activate
    cd /home/lilidl/tp_final_cv2
    python train_all.py
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")   # sin display — WSL2
import matplotlib.pyplot as plt
import torch

from src.data.dataset import make_splits, make_loaders
from src.models.backbone import DrowsinessModel
from src.training.train import train_model

H5              = "/home/lilidl/drowsiness_crops.h5"
KEEP_CLASSES    = [0, 2]
LABEL_MAP       = {0: 0, 2: 1}
CLASS_NAMES_BIN = ["alerta", "somnoliento"]
BATCH_SIZE      = 128
NUM_WORKERS     = 8

Path("checkpoints").mkdir(exist_ok=True)
Path("docs").mkdir(exist_ok=True)

# ── Splits ────────────────────────────────────────────────────────────────────
print("Preparando splits...")
train_idx, val_idx, test_idx = make_splits(H5, keep_classes=KEEP_CLASSES)
print(f"binario | train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}")
np.save("data/processed/test_idx.npy", test_idx)

train_loader, val_loader, test_loader = make_loaders(
    H5, train_idx, val_idx, test_idx,
    batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, label_map=LABEL_MAP,
)

# ── MobileNetV2 ───────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("ENTRENANDO MobileNetV2")
print("="*60)
model_a = DrowsinessModel(arch="mobilenetv2", num_classes=2, freeze=True, dropout=0.2)
print(f"Params totales:     {model_a.count_total():,}")
print(f"Entrenables fase 1: {model_a.count_trainable():,}")

history_a = train_model(
    model=model_a,
    train_loader=train_loader,
    val_loader=val_loader,
    checkpoint_path=Path("checkpoints/mobilenetv2_best.pt"),
    epochs_frozen=10,
    epochs_unfrozen=20,
    lr_head=1e-3,
    lr_finetune=1e-4,
    patience=8,
    unfreeze_blocks=2,
)
with open("checkpoints/history_mobilenetv2.json", "w") as f:
    json.dump(history_a, f, indent=2)

# ── ResNet50V2 ────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("ENTRENANDO ResNet50V2")
print("="*60)
model_b = DrowsinessModel(arch="resnet50v2", num_classes=2, freeze=True, dropout=0.2)
print(f"Params totales:     {model_b.count_total():,}")
print(f"Entrenables fase 1: {model_b.count_trainable():,}")

history_b = train_model(
    model=model_b,
    train_loader=train_loader,
    val_loader=val_loader,
    checkpoint_path=Path("checkpoints/resnet50v2_best.pt"),
    epochs_frozen=10,
    epochs_unfrozen=20,
    lr_head=1e-3,
    lr_finetune=1e-4,
    patience=8,
    unfreeze_blocks=2,
)
with open("checkpoints/history_resnet50v2.json", "w") as f:
    json.dump(history_b, f, indent=2)

# ── Curvas ────────────────────────────────────────────────────────────────────
def plot_history(ax_loss, ax_f1, history, label, color):
    epochs = range(1, len(history["train_loss"]) + 1)
    phase_changes = [i for i in range(1, len(history["phase"]))
                     if history["phase"][i] != history["phase"][i - 1]]
    ax_loss.plot(epochs, history["train_loss"], f"{color}--", alpha=0.6, label=f"{label} train")
    ax_loss.plot(epochs, history["val_loss"],   f"{color}-",              label=f"{label} val")
    ax_f1.plot(epochs,   history["train_f1"],   f"{color}--", alpha=0.6, label=f"{label} train")
    ax_f1.plot(epochs,   history["val_f1"],     f"{color}-",              label=f"{label} val")
    for pc in phase_changes:
        ax_loss.axvline(pc + 0.5, color=color, linestyle=":", alpha=0.4)
        ax_f1.axvline(pc + 0.5,   color=color, linestyle=":", alpha=0.4)

fig, (ax_loss, ax_f1) = plt.subplots(1, 2, figsize=(14, 5))
plot_history(ax_loss, ax_f1, history_a, "MobileNetV2", "b")
plot_history(ax_loss, ax_f1, history_b, "ResNet50V2",  "r")

ax_loss.set_xlabel("Epoch"); ax_loss.set_ylabel("Loss"); ax_loss.set_title("Loss")
ax_loss.legend(); ax_loss.grid(alpha=0.3)
ax_f1.set_xlabel("Epoch"); ax_f1.set_ylabel("F1-macro"); ax_f1.set_title("F1-macro")
ax_f1.legend(); ax_f1.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("docs/curvas_entrenamiento.png", dpi=150, bbox_inches="tight")
print("\nCurvas guardadas en docs/curvas_entrenamiento.png")

best_a = max(history_a["val_f1"])
best_b = max(history_b["val_f1"])
print(f"\nMejor val F1-macro:")
print(f"  MobileNetV2: {best_a:.4f}")
print(f"  ResNet50V2:  {best_b:.4f}")
