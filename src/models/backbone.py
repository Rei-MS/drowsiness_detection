"""
Modelos CNN fine-tuneados para clasificación de somnolencia (3 clases).

Soporta MobileNetV2 y ResNet50V2, ambos con pesos ImageNet preentrenados.
Fine-tuning en 2 fases:
  1. Solo la cabeza clasificadora (backbone congelado)
  2. Últimos N bloques + cabeza (backbone parcialmente descongelado)
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torchvision.models as models


NUM_CLASSES = 3


class DrowsinessModel(nn.Module):
    """
    CNN para clasificación de somnolencia en 3 clases.

    Args:
        arch: 'mobilenetv2' o 'resnet50v2'
        num_classes: número de clases de salida (default=3)
        freeze: si True, congela el backbone y solo entrena la cabeza
    """

    def __init__(
        self,
        arch: Literal["mobilenetv2", "resnet50v2"],
        num_classes: int = NUM_CLASSES,
        freeze: bool = True,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.arch = arch
        self.num_classes = num_classes

        if arch == "mobilenetv2":
            base = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V2)
            in_features = base.classifier[1].in_features
            base.classifier = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, num_classes),
            )
            self.backbone = base.features
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.head = base.classifier

        elif arch == "resnet50v2":
            base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            in_features = base.fc.in_features
            base.fc = nn.Linear(in_features, num_classes)
            # Separar backbone y cabeza para poder descongelar por bloques
            self.backbone = nn.Sequential(
                base.conv1, base.bn1, base.relu, base.maxpool,
                base.layer1, base.layer2, base.layer3, base.layer4,
            )
            self.pool = base.avgpool
            self.head = nn.Sequential(nn.Flatten(), nn.Dropout(p=dropout), base.fc)

        else:
            raise ValueError(f"arch debe ser 'mobilenetv2' o 'resnet50v2', recibido: {arch!r}")

        if freeze:
            self._freeze_backbone()

    def _freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_last_blocks(self, n: int = 2) -> None:
        """Descongela los últimos n bloques del backbone para fine-tuning fase 2."""
        if self.arch == "mobilenetv2":
            blocks = list(self.backbone.children())
            for block in blocks[-n:]:
                for param in block.parameters():
                    param.requires_grad = True

        elif self.arch == "resnet50v2":
            # backbone es Sequential: conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4
            layers = list(self.backbone.children())
            for layer in layers[-n:]:
                for param in layer.parameters():
                    param.requires_grad = True

    def unfreeze_all(self) -> None:
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        pooled = self.pool(features)
        if self.arch == "mobilenetv2":
            flat = pooled.view(pooled.size(0), -1)
            return self.head(flat)
        else:
            return self.head(pooled)

    def count_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_total(self) -> int:
        return sum(p.numel() for p in self.parameters())
