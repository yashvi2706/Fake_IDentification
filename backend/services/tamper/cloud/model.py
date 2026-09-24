"""
model.py — Multi-head ResNet18 for tamper detection.

Location: backend/services/tamper/cloud/

Architecture:
  - Shared ResNet18 backbone (pretrained ImageNet) minus final FC
  - 3 independent binary classification heads:
    - head_photo: photo_replacement (0/1)
    - head_text: text_manipulation (0/1)
    - head_compression: compression_anomaly (0/1)

Each head outputs 2 logits (authentic vs tampered) for CrossEntropyLoss.
During inference, we take softmax[:, 1] as the "tampered" probability per head.
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


class MultiHeadResNet18(nn.Module):
    """
    ResNet18 backbone with 3 binary classification heads.
    
    Forward returns dict of logits:
    {
        'photo_replacement': (B, 2),
        'text_manipulation': (B, 2),
        'compression_anomaly': (B, 2)
    }
    """

    def __init__(self, pretrained=True):
        super().__init__()

        # Load backbone
        if pretrained:
            backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            backbone = models.resnet18(weights=None)

        # Remove the final FC layer — keep everything up to avgpool
        self.features = nn.Sequential(*list(backbone.children())[:-1])

        # Feature dimension from ResNet18 = 512
        feat_dim = 512

        # 3 independent binary classification heads
        self.head_photo = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
        )

        self.head_text = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
        )

        self.head_compression = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        # Shared feature extraction
        features = self.features(x)       # (B, 512, 1, 1)
        features = features.flatten(1)    # (B, 512)

        return {
            "photo_replacement": self.head_photo(features),
            "text_manipulation": self.head_text(features),
            "compression_anomaly": self.head_compression(features),
        }

    def predict_probabilities(self, x):
        """
        Inference helper — returns sigmoid/softmax probabilities for each head.
        Returns dict of (B,) tensors with tampered probability.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = {}
            for head_name, head_logits in logits.items():
                p = torch.nn.functional.softmax(head_logits, dim=1)[:, 1]
                probs[head_name] = p
            return probs


def masked_cross_entropy_loss(logits_dict, labels_dict):
    """
    Compute CrossEntropyLoss only on samples where label != -1 (not masked).
    
    Args:
        logits_dict: dict of head_name -> (B, 2) logits
        labels_dict: dict of head_name -> (B,) LongTensor with values 0, 1, or -1
    
    Returns:
        total_loss: scalar tensor (mean of all non-masked losses)
        per_head_loss: dict of head_name -> scalar loss (or 0 if all masked)
    """
    criterion = nn.CrossEntropyLoss(reduction="mean")
    total_loss = torch.tensor(0.0, device=next(iter(logits_dict.values())).device, requires_grad=True)
    per_head_loss = {}
    num_active_heads = 0

    for head_name in logits_dict:
        logits = logits_dict[head_name]      # (B, 2)
        labels = labels_dict[head_name]      # (B,)

        # Mask: only keep samples where label != -1
        mask = labels != -1
        if mask.sum() == 0:
            per_head_loss[head_name] = 0.0
            continue

        masked_logits = logits[mask]
        masked_labels = labels[mask]

        head_loss = criterion(masked_logits, masked_labels)
        per_head_loss[head_name] = head_loss.item()
        total_loss = total_loss + head_loss
        num_active_heads += 1

    if num_active_heads > 0:
        total_loss = total_loss / num_active_heads

    return total_loss, per_head_loss


def compute_per_head_accuracy(logits_dict, labels_dict):
    """
    Compute accuracy per head, ignoring masked samples.
    Returns dict of head_name -> accuracy (float) or None if all masked.
    """
    accuracies = {}
    for head_name in logits_dict:
        logits = logits_dict[head_name]
        labels = labels_dict[head_name]

        mask = labels != -1
        if mask.sum() == 0:
            accuracies[head_name] = None
            continue

        preds = logits[mask].argmax(dim=1)
        correct = (preds == labels[mask]).float().mean().item()
        accuracies[head_name] = correct

    return accuracies
