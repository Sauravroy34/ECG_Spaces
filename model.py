"""
ResNet50 for 1D Data
====================

A standard ResNet-50 architecture adapted for 1-dimensional input
(e.g. time-series, spectral data, audio waveforms, sensor signals).

All 2D operations (Conv2d, BatchNorm2d, AdaptiveAvgPool2d) are replaced
with their 1D counterparts (Conv1d, BatchNorm1d, AdaptiveAvgPool1d).

Last Hidden State Dimensions
-----------------------------
After the global average pooling and before the final FC layer, the
feature tensor (the "last hidden state") has shape:

    (batch_size, 2048)

This is because the final bottleneck block outputs 512 * 4 = 2048
channels, and AdaptiveAvgPool1d(1) collapses the temporal dimension
to 1, which is then flattened to give a 2048-d vector per sample.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Type


# ======================================================================
#  Class labels for the 6-class ECG arrhythmia task
# ======================================================================

CLASS_NAMES: List[str] = ["ATRIAL", "SA", "SB", "SR", "ST", "SVT"]

CLASS_FULL_NAMES: dict[str, str] = {
    "ATRIAL": "Atrial Arrhythmia",
    "SA": "Sinus Arrhythmia",
    "SB": "Sinus Bradycardia",
    "SR": "Sinus Rhythm",
    "ST": "Sinus Tachycardia",
    "SVT": "Supraventricular Tachycardia",
}

NUM_CLASSES: int = len(CLASS_NAMES)


class Bottleneck1d(nn.Module):
    """
    Bottleneck building block for ResNet-50 (1D variant).

    Each bottleneck has three conv layers:
        1×1  (reduce channels)  ->  3×1 (spatial)  ->  1×1 (expand channels)

    The expansion factor is 4, so the output channels are
    `planes * expansion`.
    """

    expansion: int = 4

    def __init__(
        self,
        in_channels: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()

        out_channels = planes * self.expansion

        # 1×1 conv  –  reduce channels
        self.conv1 = nn.Conv1d(in_channels, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)

        # 3×1 conv  –  spatial convolution (stride applied here)
        self.conv2 = nn.Conv1d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm1d(planes)

        # 1×1 conv  –  expand channels
        self.conv3 = nn.Conv1d(planes, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet50_1d(nn.Module):
    """
    ResNet-50 for 1-dimensional inputs.

    Architecture (layer counts = [3, 4, 6, 3]):
        ┌─────────────────────────────────────────────────────┐
        │  Conv1d(in_channels, 64, k=7, s=2, p=3)            │
        │  BatchNorm1d(64)  →  ReLU  →  MaxPool1d(k=3,s=2,p=1)│
        ├─────────────────────────────────────────────────────┤
        │  Layer 1:  3 × Bottleneck(64  → 256)                │
        │  Layer 2:  4 × Bottleneck(128 → 512),   stride=2    │
        │  Layer 3:  6 × Bottleneck(256 → 1024),  stride=2    │
        │  Layer 4:  3 × Bottleneck(512 → 2048),  stride=2    │
        ├─────────────────────────────────────────────────────┤
        │  AdaptiveAvgPool1d(1)                                │
        │  Flatten  →  (batch_size, 2048)   ← last hidden state│
        │  Linear(2048, num_classes)                           │
        └─────────────────────────────────────────────────────┘

    Parameters
    ----------
    in_channels : int
        Number of input channels (e.g. 1 for univariate time-series,
        or the number of sensor channels / spectral bands).
    num_classes : int
        Number of output classes for the final fully-connected layer.
        Set to 0 or None to skip the FC head and return the 2048-d
        feature vector directly (useful as a backbone / encoder).
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 1000,
        return_last_hidden_state: bool = False
    ) -> None:
        super().__init__()

        self.return_last_hidden_state = return_last_hidden_state
        self.in_channels_current = 64
        layers: List[int] = [3, 4, 6, 3]  # ResNet-50 config
        
        # ── Stem ──────────────────────────────────────────────
        self.conv1 = nn.Conv1d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # ── Residual stages ──────────────────────────────────
        self.layer1 = self._make_layer(planes=64,  num_blocks=layers[0], stride=1)
        self.layer2 = self._make_layer(planes=128, num_blocks=layers[1], stride=2)
        self.layer3 = self._make_layer(planes=256, num_blocks=layers[2], stride=2)
        self.layer4 = self._make_layer(planes=512, num_blocks=layers[3], stride=2)

        # ── Head ─────────────────────────────────────────────
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.num_classes = num_classes
        if num_classes and num_classes > 0:
            self.fc = nn.Linear(512 * Bottleneck1d.expansion, num_classes)
        else:
            self.fc = nn.Identity()

        # ── Weight initialisation ────────────────────────────
        self._init_weights()

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------

    def _make_layer(
        self,
        planes: int,
        num_blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        """Build one residual stage."""
        downsample = None
        out_channels = planes * Bottleneck1d.expansion

        # Downsample shortcut when spatial size or channel count changes
        if stride != 1 or self.in_channels_current != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(
                    self.in_channels_current,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(out_channels),
            )

        blocks: List[nn.Module] = []
        # First block may downsample
        blocks.append(
            Bottleneck1d(self.in_channels_current, planes, stride, downsample)
        )
        self.in_channels_current = out_channels

        # Remaining blocks
        for _ in range(1, num_blocks):
            blocks.append(Bottleneck1d(self.in_channels_current, planes))

        return nn.Sequential(*blocks)

    def _init_weights(self) -> None:
        """Kaiming initialisation (standard for ResNets)."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    # ------------------------------------------------------------------
    #  Forward pass
    # ------------------------------------------------------------------

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract the last hidden state (2048-d feature vector).

        Returns
        -------
        torch.Tensor
            Shape: (batch_size, 2048)
        """
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)          # (B, 2048, 1)
        x = torch.flatten(x, 1)     # (B, 2048)  ← last hidden state
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape (batch_size, in_channels, length).

        Returns
        -------
        torch.Tensor
            If num_classes > 0:  (batch_size, num_classes)
            Otherwise:           (batch_size, 2048)
        """
        x = self.get_features(x) # (B, 2048)
        
        if self.return_last_hidden_state:
            return x

        x = self.fc(x)
        return x


# ======================================================================
#  Convenience constructor
# ======================================================================

def resnet50_1d(
    in_channels: int = 1,
    num_classes: int = 1000,
    return_last_hidden_state: bool = False
) -> ResNet50_1d:
    """
    Build a ResNet-50 for 1D data.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    num_classes : int
        Number of output classes (set 0 to use as a feature extractor).

    Returns
    -------
    ResNet50_1d
    """
    return ResNet50_1d(
        in_channels=in_channels,
        num_classes=num_classes,
        return_last_hidden_state=return_last_hidden_state,
    )


# ======================================================================
#  ECG classifier wrapper
# ======================================================================

class ECG_1D(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()

        self.model = resnet50_1d(
            in_channels=1,
            return_last_hidden_state=True,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),

            nn.Linear(2048, 1024),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),

            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        x = self.classifier(x)
        return x
