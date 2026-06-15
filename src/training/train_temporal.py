"""
Loop de entrenamiento para TemporalDrowsinessModel.

Fase 1 (epochs_frozen):   solo GRU + attention + head, backbone congelado
Fase 2 (epochs_unfrozen): descongelar últimos N bloques del backbone
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

from src.models.temporal_model import TemporalDrowsinessModel


def _run_epoch(
    model: TemporalDrowsinessModel,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    train: bool,
    scaler: Optional[GradScaler] = None,
) -> tuple[float, float]:
    model.train(train)
    total_loss = 0.0
    all_preds, all_labels = [], []
    use_amp = scaler is not None and device.type == "cuda"

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for seqs, labels in loader:
            seqs   = seqs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if train:
                optimizer.zero_grad()

            with autocast(device_type=device.type, enabled=use_amp):
                logits = model(seqs)
                loss   = criterion(logits, labels)

            if train:
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

            total_loss += loss.item() * len(labels)
            all_preds.append(logits.argmax(1).cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds  = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    return (total_loss / len(all_labels),
            f1_score(all_labels, all_preds, average="macro", zero_division=0))


def train_temporal(
    model: TemporalDrowsinessModel,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    checkpoint_path: Path,
    epochs_frozen:   int = 10,
    epochs_unfrozen: int = 25,
    lr_head:         float = 1e-3,
    lr_finetune:     float = 5e-5,
    patience:        int = 8,
    unfreeze_blocks: int = 2,
    device: Optional[torch.device] = None,
) -> dict:
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

    def _run_phase(epochs, lr, phase_name, early_stop=True):
        nonlocal best_val_f1
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, weight_decay=1e-4,
        )
        scheduler = ReduceLROnPlateau(optimizer, mode="max",
                                      patience=patience // 2, factor=0.5)

        print(f"\n=== {phase_name} | {epochs} epochs | lr={lr} | "
              f"params entrenables: {model.count_trainable():,}/{model.count_total():,} ===")

        phase_best, no_improve = -1.0, 0

        for epoch in range(1, epochs + 1):
            t_loss, t_f1 = _run_epoch(model, train_loader, criterion, optimizer,
                                      device, True,  scaler)
            v_loss, v_f1 = _run_epoch(model, val_loader,   criterion, None,
                                      device, False, None)
            scheduler.step(v_f1)

            history["train_loss"].append(t_loss)
            history["val_loss"].append(v_loss)
            history["train_f1"].append(t_f1)
            history["val_f1"].append(v_f1)
            history["phase"].append(phase_name)

            print(f"  Epoch {epoch:02d}/{epochs}  "
                  f"train_loss={t_loss:.4f}  val_loss={v_loss:.4f}  "
                  f"train_f1={t_f1:.4f}  val_f1={v_f1:.4f}")

            if v_f1 > best_val_f1:
                best_val_f1 = v_f1
                torch.save(
                    {"model_state": model.state_dict(), "val_f1": v_f1,
                     "arch": model.arch, "num_classes": 2, "seq_len": model.seq_len},
                    checkpoint_path,
                )
                print(f"    ✓ Mejor checkpoint guardado (val_f1={v_f1:.4f})")

            if v_f1 > phase_best:
                phase_best, no_improve = v_f1, 0
            else:
                no_improve += 1
                if early_stop and no_improve >= patience:
                    print(f"  Early stopping tras {patience} epochs sin mejora.")
                    return True
        return False

    _run_phase(epochs_frozen,   lr_head,     "fase1_temporal", early_stop=False)
    model.unfreeze_last_blocks(n=unfreeze_blocks)
    _run_phase(epochs_unfrozen, lr_finetune, "fase2_finetune", early_stop=True)

    print(f"\nEntrenamiento finalizado. Mejor val_f1={best_val_f1:.4f}")
    print(f"Checkpoint: {checkpoint_path}")
    return history
