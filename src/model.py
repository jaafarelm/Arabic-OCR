"""
model.py — ResNet-style CNN in PyTorch for Arabic isolated-character recognition.

Ported from Keras to PyTorch so training can run on the RTX 5070 Ti (Blackwell,
sm_120), which TensorFlow does not support.

Architecture is the same ResNet pattern as before:
    stem conv -> 3 residual stages (64 -> 128 -> 256) -> global avg pool -> FC

Why residual blocks:
    Each block learns a RESIDUAL correction added to its own input via a skip
    connection:  output = input + F(input).  This keeps gradients flowing
    through deep stacks so the network trains reliably.

PyTorch differences from Keras to note while reading:
    - You define layers in __init__ and wire them in forward().
    - There is no .compile(); loss and optimizer live in the training script.
    - Conv2d takes (in_channels, out_channels), so each layer must know its
      input channel count explicitly.
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Two 3x3 convs plus a skip connection.

    If downsample=True the block halves the spatial size (stride 2) and the
    shortcut uses a 1x1 conv so its shape matches the main path.
    """

    def __init__(self, in_channels, out_channels, downsample=False):
        super().__init__()
        stride = 2 if downsample else 1

        # --- Main path: Conv -> BN -> ReLU -> Conv -> BN ---
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels, momentum=0.1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)

        # --- Shortcut: match shape only if we downsampled or changed channels ---
        if downsample or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels, momentum=0.1),
            )
        else:
            self.shortcut = nn.Identity()   # pass the input through unchanged

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out = out + identity      # the skip connection
        return self.relu(out)


class ResNetCNN(nn.Module):
    """ResNet-style classifier for 32x32 single-channel character images."""

    def __init__(self, num_classes, in_channels=1):
        super().__init__()

        # --- Stem: initial conv before the residual stages ---
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=0.1),
            nn.ReLU(inplace=True),
        )

        # --- Residual stages: channels grow as spatial size shrinks ---
        self.stage1 = nn.Sequential(          # 32x32, 64 channels
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
        )
        self.stage2 = nn.Sequential(          # -> 16x16, 128 channels
            ResidualBlock(64, 128, downsample=True),
            ResidualBlock(128, 128),
        )
        self.stage3 = nn.Sequential(          # -> 8x8, 256 channels
            ResidualBlock(128, 256, downsample=True),
            ResidualBlock(256, 256),
        )

        # --- Head ---
        # Global average pooling collapses each feature map to one number:
        # far fewer parameters than flatten+dense, and less overfitting.
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)

        x = self.pool(x)              # (batch, 256, 1, 1)
        x = torch.flatten(x, 1)       # (batch, 256)
        x = self.dropout(x)
        # NOTE: no softmax here — PyTorch's CrossEntropyLoss expects raw logits.
        return self.fc(x)


def build_model(num_classes, in_channels=1):
    """Factory kept for parity with the old Keras API."""
    return ResNetCNN(num_classes=num_classes, in_channels=in_channels)


if __name__ == "__main__":
    m = build_model(num_classes=115)
    n_params = sum(p.numel() for p in m.parameters())
    print(m)
    print(f"\nTotal parameters: {n_params:,}")

    # Shape check: a batch of 4 grayscale 32x32 images -> 115 class scores.
    dummy = torch.randn(4, 1, 32, 32)
    print("Output shape:", m(dummy).shape)   # expect torch.Size([4, 115])
