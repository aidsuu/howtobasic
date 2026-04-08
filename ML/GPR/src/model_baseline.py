#!/usr/bin/env python3
"""Small CNN encoder-decoder baseline for MERL-GPR inversion."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """A minimal convolutional block for stable baseline experiments."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SmallEncoderDecoder(nn.Module):
    """A compact encoder-decoder that preserves the final spatial size."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        base_channels: int = 32,
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError(f"in_channels must be > 0, got {in_channels}")
        if out_channels <= 0:
            raise ValueError(f"out_channels must be > 0, got {out_channels}")

        self.enc1 = ConvBlock(in_channels, base_channels)
        self.down1 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1)

        self.enc2 = ConvBlock(base_channels * 2, base_channels * 2)
        self.down2 = nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1)

        self.bottleneck = ConvBlock(base_channels * 4, base_channels * 4)

        self.dec2 = ConvBlock(base_channels * 4, base_channels * 2)
        self.dec1 = ConvBlock(base_channels * 2, base_channels)
        self.head = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_spatial_size = x.shape[-2:]

        x1 = self.enc1(x)
        x2 = self.enc2(self.down1(x1))
        xb = self.bottleneck(self.down2(x2))

        xu = F.interpolate(xb, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        xu = self.dec2(xu)

        xu = F.interpolate(xu, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        xu = self.dec1(xu)

        out = self.head(xu)
        if out.shape[-2:] != input_spatial_size:
            out = F.interpolate(out, size=input_spatial_size, mode="bilinear", align_corners=False)
        return out

