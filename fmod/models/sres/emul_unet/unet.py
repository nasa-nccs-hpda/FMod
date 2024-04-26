import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Tuple, Type, Optional, Union, Sequence, Mapping

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

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


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # input is CHW
        diffy: int = x2.size()[2] - x1.size()[2]
        diffx: int = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffx // 2, diffx - diffx // 2,  diffy // 2, diffy - diffy // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x: torch.Tensor = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class Upscale(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels: int, upscale_fator: int, bilinear: bool = False):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=upscale_fator, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, in_channels )
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=upscale_fator)
            self.conv = DoubleConv(in_channels, in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels: int, upscale_factors: List[int], bilinear: bool=False):
        super(UNet, self).__init__()
        self.n_channels: int = n_channels
        self.bilinear: bool = bilinear

        self.inc: nn.Module = DoubleConv(n_channels, 64)
        self.down1: nn.Module = Down(64, 128)
        self.down2: nn.Module = Down(128, 256)
        self.down3: nn.Module = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4: nn.Module = Down(512, 1024 // factor)
        self.up1: nn.Module = Up(1024, 512 // factor, bilinear)
        self.up2: nn.Module = Up(512, 256 // factor, bilinear)
        self.up3: nn.Module = Up(256, 128 // factor, bilinear)
        self.up4: nn.Module = Up(128, 64, bilinear)
        self.upscale_layers = nn.ModuleList([ Upscale(64, usf, bilinear) for usf in upscale_factors] )
        self.upscale = nn.Sequential( self.upscale_layers )
        self.outc: nn.Module = OutConv(64, self.n_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.upscale(x)
        result = self.outc(x)
        return result