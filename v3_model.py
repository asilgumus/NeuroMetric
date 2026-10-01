"""SFCN brain-age model adapted from the released UK Biobank architecture.

The feature extractor and 40-class checkpoint are attributable to
https://github.com/ha-ha-ha-han/UKBiobank_deep_pretrain (MIT license).
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import math

import torch
from torch import nn


AGE_MIN, AGE_MAX = 18, 100
N_CLASSES = AGE_MAX - AGE_MIN + 1


class SFCN(nn.Module):
    def __init__(self, n_classes: int = N_CLASSES):
        super().__init__()
        blocks = OrderedDict()
        previous = 1
        for index, channels in enumerate((32, 64, 128, 256, 256)):
            blocks[f"conv_{index}"] = nn.Sequential(
                nn.Conv3d(previous, channels, 3, padding=1),
                nn.BatchNorm3d(channels), nn.MaxPool3d(2, 2), nn.ReLU(inplace=True))
            previous = channels
        blocks["conv_5"] = nn.Sequential(
            nn.Conv3d(256, 64, 1), nn.BatchNorm3d(64), nn.ReLU(inplace=True))
        self.feature_extractor = nn.Sequential(blocks)
        self.classifier = nn.Sequential(OrderedDict([
            ("average_pool", nn.AvgPool3d((5, 6, 5))),
            ("dropout", nn.Dropout(0.5)),
            ("conv_6", nn.Conv3d(64, n_classes, 1)),
        ]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.classifier(self.feature_extractor(x)).flatten(1)
        return logits


def load_pretrained(path: Path, device: str | torch.device = "cpu") -> SFCN:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = {key.removeprefix("module."): value for key, value in checkpoint.items()}
    source = SFCN(40)
    source.load_state_dict(state, strict=True)
    target = SFCN()
    target.feature_extractor.load_state_dict(source.feature_extractor.state_dict(), strict=True)
    # UKB checkpoint has bins 42..81; keep learned logits for those ages.
    with torch.no_grad():
        target.classifier.conv_6.weight.zero_()
        # An all-zero bias makes the 43 new bins compete with the pretrained
        # logits before they have seen any data, collapsing initial estimates
        # toward the middle of the enlarged 18..100 range. Keep them trainable,
        # but give the transferred UKB head a meaningful starting prior.
        target.classifier.conv_6.bias.fill_(-3.)
        low, high = 42 - AGE_MIN, 82 - AGE_MIN
        target.classifier.conv_6.weight[low:high].copy_(source.classifier.conv_6.weight)
        target.classifier.conv_6.bias[low:high].copy_(source.classifier.conv_6.bias)
    return target.to(device)


def expected_age(logits: torch.Tensor) -> torch.Tensor:
    centers = torch.arange(AGE_MIN, AGE_MAX + 1, device=logits.device,
                           dtype=logits.dtype)
    return (logits.softmax(1) * centers).sum(1)


def precision_loss(prediction: torch.Tensor, age: torch.Tensor) -> torch.Tensor:
    """Bounded Welsch auxiliary loss; fixed half-year scale, continuous labels."""
    if (age.ndim != 1 or prediction.shape != age.shape or age.numel() == 0
            or not torch.isfinite(age).all() or not torch.isfinite(prediction).all()
            or ((age < AGE_MIN) | (age > AGE_MAX)).any()):
        raise ValueError("Precision loss requires aligned finite chronological ages")
    return (-torch.expm1(-.5 * ((prediction - age) / .5).square())).mean()


def gaussian_targets(age: torch.Tensor, sigma: float = 2.) -> torch.Tensor:
    if not math.isfinite(sigma) or not .5 <= sigma <= 2.:
        raise ValueError("Target sigma must be finite and in [0.5, 2]")
    if age.ndim != 1 or not torch.isfinite(age).all() or ((age < AGE_MIN) | (age > AGE_MAX)).any():
        raise ValueError("Targets require finite chronological ages in [18, 100]")
    centers = torch.arange(AGE_MIN, AGE_MAX + 1, device=age.device,
                           dtype=age.dtype)
    weights = torch.exp(-0.5 * ((centers[None] - age[:, None]) / sigma) ** 2)
    return weights / weights.sum(1, keepdim=True)


def rounded_interval_loss(prediction: torch.Tensor, age: torch.Tensor) -> torch.Tensor:
    """Smooth rounded-hit surrogate; labels stay continuous in KL and MAE.

    Logistic window centered on the ties-to-even rounded label, temperature
    0.25 years. Exact boundary inclusion is evaluated only with np.rint.
    """
    if (age.ndim != 1 or prediction.shape != age.shape or age.numel() == 0
            or not torch.isfinite(age).all() or not torch.isfinite(prediction).all()
            or ((age < AGE_MIN) | (age > AGE_MAX)).any()):
        raise ValueError("Interval loss requires aligned finite chronological ages")
    center = torch.round(age)
    inside = (torch.sigmoid((prediction - (center - .5)) / .25)
              - torch.sigmoid((prediction - (center + .5)) / .25))
    return (1 - inside).mean()


def rounded_interval_nll(prediction: torch.Tensor, age: torch.Tensor) -> torch.Tensor:
    """Stable quarter-year-scaled negative log logistic interval probability.

    Same rounded interval as rounded_interval_loss, with nonvanishing distant
    prediction gradients bounded by one. Only a train-time surrogate.
    """
    if (age.ndim != 1 or prediction.shape != age.shape or age.numel() == 0
            or not torch.isfinite(age).all() or not torch.isfinite(prediction).all()
            or ((age < AGE_MIN) | (age > AGE_MAX)).any()):
        raise ValueError("Interval NLL requires aligned finite chronological ages")
    center = torch.round(age)
    lower = (prediction - (center - .5)) / .25
    upper = (prediction - (center + .5)) / .25
    # log(sigmoid(lower)-sigmoid(upper)) without subtracting saturated CDFs.
    log_width = math.log(-math.expm1(-4.))
    return .25 * (nn.functional.softplus(-lower)
                  + nn.functional.softplus(upper) - log_width).mean()
