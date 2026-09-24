"""Compact multi-signal tamper model for Colab T4 training."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from torchvision import models


class CompactTamperNet(nn.Module):
    """
    Compact dual-head model for ID tamper detection.

    Input: 6 channels
      - RGB (3)
      - ELA (3)

    Heads:
      - photo_replacement: face-photo splice
      - document_tamper: non-face/global tamper
    """

    def __init__(self, pretrained: bool = True, freeze_backbone: bool = True):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)

        conv1_w = backbone.conv1.weight.data.clone()
        backbone.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            backbone.conv1.weight[:, :3] = conv1_w
            backbone.conv1.weight[:, 3:] = conv1_w

        self.backbone = nn.Sequential(*list(backbone.children())[:-1])

        feat_dim = 512
        self.photo_head = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(128, 2),
        )
        self.doc_head = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(128, 2),
        )

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x6: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.backbone(x6).flatten(1)
        return {
            "photo_replacement": self.photo_head(feat),
            "document_tamper": self.doc_head(feat),
        }


class MultiTaskCrossEntropy(nn.Module):
    """Masked CE: labels can be -1 for unavailable head."""

    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, logits: Dict[str, torch.Tensor], labels: Dict[str, torch.Tensor]):
        total = None
        details = {}
        n = 0
        for head, pred in logits.items():
            y = labels[head]
            m = y != -1
            if m.sum() == 0:
                details[head] = 0.0
                continue
            loss = self.criterion(pred[m], y[m])
            total = loss if total is None else total + loss
            details[head] = float(loss.item())
            n += 1

        if total is None:
            return torch.tensor(0.0, device=next(iter(logits.values())).device, requires_grad=True), details
        return total / max(1, n), details
