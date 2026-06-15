"""
Loop de entrenamiento para DrowsinessModel.

Fine-tuning en 2 fases:
  Fase 1 (epochs_frozen):  solo la cabeza, backbone congelado
  Fase 2 (epochs_unfrozen): últimos N bloques + cabeza, lr reducido

Patrón de Clase_1/CIFAR10_CNN.ipynb y Clase_3 (pretrained models).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.backbone import DrowsinessModel


def _run_epoch(
    model: DrowsinessModel,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    train: bool,
    scaler: Optional[GradScaler] = None,
) -> tuple[float, float]:
    """Ejecuta una epoch de train o validación. Retorna (loss_media, f1_macro)."""
    model.train(train)
    total_loss = 0.0
    all_preds, all_labels = [], []
    use_amp = scaler is not None and device.type == "cuda"

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if train:
                optimizer.zero_grad()

            with autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)

            if train:
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

            total_loss += loss.item() * len(labels)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())

    all_preds  = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    avg_loss = total_loss / len(all_labels)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, f1


def train_model(
    model: DrowsinessModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    checkpoint_path: Path,
    epochs_frozen: int = 10,
    epochs_unfrozen: int = 20,
    lr_head: float = 1e-3,
    lr_finetune: float = 1e-4,
    patience: int = 5,
    unfreeze_blocks: int = 2,
    device: Optional[torch.device] = None,
) -> dict:
    """
    Entrena el modelo en 2 fases y guarda el mejor checkpoint por val_f1_macro.

    Retorna:
        history dict con listas: train_loss, val_loss, train_f1, val_f1, phase
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}  |  AMP: {device.type == 'cuda'}")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    scaler = GradScaler(device=device.type) if device.type == "cuda" else None

    history = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": [], "phase": []}
    best_val_f1 = -1.0

    def _run_phase(
        epochs: int,
        lr: float,
        phase_name: str,
        early_stop: bool = True,
    ) -> bool:
        nonlocal best_val_f1
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4
        )
        scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=patience // 2, factor=0.5)

        trainable = model.count_trainable()
        total = model.count_total()
        print(f"\n=== {phase_name} | {epochs} epochs | lr={lr} | params entrenables: {trainable:,}/{total:,} ===")

        phase_best = -1.0
        no_improve = 0

        for epoch in range(1, epochs + 1):
            t_loss, t_f1 = _run_epoch(model, train_loader, criterion, optimizer, device, train=True,  scaler=scaler)
            v_loss, v_f1 = _run_epoch(model, val_loader,   criterion, None,      device, train=False, scaler=None)
            scheduler.step(v_f1)

            history["train_loss"].append(t_loss)
            history["val_loss"].append(v_loss)
            history["train_f1"].append(t_f1)
            history["val_f1"].append(v_f1)
            history["phase"].append(phase_name)

            print(
                f"  Epoch {epoch:02d}/{epochs}  "
                f"train_loss={t_loss:.4f}  val_loss={v_loss:.4f}  "
                f"train_f1={t_f1:.4f}  val_f1={v_f1:.4f}"
            )

            # Checkpoint: siempre guarda el mejor GLOBAL (cruza ambas fases)
            if v_f1 > best_val_f1:
                best_val_f1 = v_f1
                torch.save(
                    {"model_state": model.state_dict(), "val_f1": v_f1, "arch": model.arch,
                     "num_classes": model.num_classes},
                    checkpoint_path,
                )
                print(f"    ✓ Mejor checkpoint guardado (val_f1={v_f1:.4f})")

            # Early stopping: cuenta epochs sin mejora DENTRO de la fase
            if v_f1 > phase_best:
                phase_best = v_f1
                no_improve = 0
            else:
                no_improve += 1
                if early_stop and no_improve >= patience:
                    print(f"  Early stopping tras {patience} epochs sin mejora.")
                    return True  # stop
        return False

    # Fase 1 (warmup de la cabeza): corre completa SIN early stopping, para no
    # abortar antes del fine-tuning — el backbone congelado se estanca de entrada.
    _run_phase(epochs_frozen, lr_head, "fase1_cabeza", early_stop=False)

    # Fase 2: descongelar últimos bloques. SIEMPRE se ejecuta.
    model.unfreeze_last_blocks(n=unfreeze_blocks)
    _run_phase(epochs_unfrozen, lr_finetune, "fase2_finetune", early_stop=True)

    print(f"\nEntrenamiento finalizado. Mejor val_f1={best_val_f1:.4f}")
    print(f"Checkpoint: {checkpoint_path}")
    return history
