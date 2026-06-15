"""
TemporalDrowsinessModel — ResNet50V2 (feature extractor) + BiGRU + Attention.

Arquitectura:
  CNN backbone (congelado inicialmente) → feature por frame (2048-d)
  → LayerNorm → BiGRU (512 hidden, 2 capas) → Attention pooling → head
"""

from __future__ import annotations
from pathlib import Path

import torch
import torch.nn as nn

from src.models.backbone import DrowsinessModel

FEAT_SIZE = {"mobilenetv2": 1280, "resnet50v2": 2048}


class _AttentionPool(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, dim)
        w = torch.softmax(self.fc(x), dim=1)   # (B, T, 1)
        return (w * x).sum(dim=1)               # (B, dim)


class TemporalDrowsinessModel(nn.Module):
    """
    Args:
        backbone_checkpoint: path al .pt del CNN entrenado (resnet50v2_best.pt)
        seq_len:  longitud de secuencia (informativo, no restringe el forward)
        hidden:   tamaño del estado oculto GRU por dirección
        layers:   capas GRU apiladas
        dropout:  dropout en GRU y cabeza
        num_classes: 2 para tarea binaria
    """

    def __init__(
        self,
        backbone_checkpoint: str | Path,
        seq_len: int = 16,
        hidden: int = 512,
        layers: int = 2,
        dropout: float = 0.3,
        num_classes: int = 2,
    ):
        super().__init__()
        self.seq_len = seq_len

        # ── Cargar backbone CNN ───────────────────────────────────────────────
        ckpt = torch.load(backbone_checkpoint, map_location="cpu", weights_only=True)
        arch  = ckpt["arch"]
        n_cls = ckpt["num_classes"]

        base = DrowsinessModel(arch=arch, num_classes=n_cls, freeze=False)
        base.load_state_dict(ckpt["model_state"])

        self.arch      = arch
        self.backbone  = base.backbone
        self.pool      = base.pool
        self.feat_size = FEAT_SIZE[arch]
        self._freeze_backbone()

        # ── Componente temporal ───────────────────────────────────────────────
        self.frame_drop = nn.Dropout(p=0.1)  # zeroa frames enteros durante train
        self.norm = nn.LayerNorm(self.feat_size)
        self.gru  = nn.GRU(
            input_size    = self.feat_size,
            hidden_size   = hidden,
            num_layers    = layers,
            batch_first   = True,
            bidirectional = True,
            dropout       = dropout if layers > 1 else 0.0,
        )
        self.attn = _AttentionPool(hidden * 2)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes),
        )

    # ── Freeze / unfreeze ─────────────────────────────────────────────────────
    def _freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.pool.parameters():
            p.requires_grad = False

    def unfreeze_last_blocks(self, n: int = 2) -> None:
        layers = list(self.backbone.children())
        for layer in layers[-n:]:
            for p in layer.parameters():
                p.requires_grad = True

    def count_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_total(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # ── Forward ───────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C, H, W) → logits (B, num_classes)"""
        B, T, C, H, W = x.shape

        feats  = self.backbone(x.view(B * T, C, H, W))     # (B*T, feat, h, w)
        pooled = self.pool(feats).flatten(1)                # (B*T, feat_size)
        pooled = pooled.view(B, T, self.feat_size)          # (B, T, feat_size)
        pooled = self.norm(pooled)
        pooled = self.frame_drop(pooled)   # temporal dropout (zeroa features completos)

        gru_out, _ = self.gru(pooled)                       # (B, T, hidden*2)
        context    = self.attn(gru_out)                     # (B, hidden*2)

        return self.head(context)
