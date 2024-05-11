import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Tuple, Type, Optional, Union, Sequence, Mapping
from fmod.base.util.logging import lgm, exception_handled, log_timing

class DoubleConv(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, mid_channels: int=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding='same', bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding='same', bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)

class Downscale(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.maxpool_conv(x)
        return y

class Upscale(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv( 2*out_channels, out_channels )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        xup = self.up(x)
        y: torch.Tensor = torch.cat([xup, skip], dim=1 )
        return self.conv(y)


class UNet(nn.Module):
    def __init__(self, nfeatures: int, depth: int ):
        super(UNet, self).__init__()
        self.depth: int = depth
        self.downscale = nn.ModuleList()
        self.upscale = nn.ModuleList()

        for iL in range(depth):
            usf, dsf = 2 ** (depth-iL-1), 2 ** iL
            self.downscale.append( Downscale(nfeatures * dsf, nfeatures * dsf * 2))
            self.upscale.append( Upscale(nfeatures * usf * 2, nfeatures * usf))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = []
        for iL in range(self.depth):
            skip.insert(0, x)
            x: torch.Tensor = self.downscale[iL](x)
        for iL in range(self.depth):
            x = self.upscale[iL](x,skip[iL])
        return x


