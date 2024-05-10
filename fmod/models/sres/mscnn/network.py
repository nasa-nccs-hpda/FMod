import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Tuple, Type, Optional, Union, Sequence, Mapping
from fmod.base.util.logging import lgm, exception_handled, log_timing

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels: int, out_channels: int, mid_channels: int=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding='same', bias=True),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(negative_slope = 0.1),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding='same', bias=True),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope = 0.1),
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
        y = self.maxpool_conv(x)
        return y

class Upscale(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels: int, out_channels: int, upscale_fator: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=upscale_fator)
        self.conv = DoubleConv(out_channels, out_channels )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        return self.conv(x)

class Upsample(nn.Module):

    def __init__(self, upscale_fator: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=upscale_fator, mode='bilinear', align_corners=False )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)

class Crossscale(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(Crossscale, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class MSCNN(nn.Module):
    def __init__(self, n_channels: int, nfeatures: int, upscale_factors: List[int], bilinear: bool=False ):
        super(MSCNN, self).__init__()
        self.n_channels: int = n_channels
        self.upscale_factors = upscale_factors
        self.inc: nn.Module = DoubleConv( n_channels, nfeatures )
        self.upscale: nn.ModuleList = nn.ModuleList()
        self.upsample: nn.ModuleList = nn.ModuleList()
        self.crossscale: nn.ModuleList = nn.ModuleList()
        for iL, usf in enumerate(upscale_factors):
            in_channels = nfeatures if iL == 0 else nfeatures*2
            self.upscale.append(  Upscale( in_channels, nfeatures*2, usf ) )
            self.crossscale.append(  Crossscale( nfeatures*2, self.n_channels ) )
            self.upsample.append( Upsample(usf) )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features, results = self.inc(x), [x]
        for iL, usf in enumerate(self.upscale_factors):
            features = self.upscale[iL](features)
            xave = self.upsample[iL](results[-1])
            xres = self.crossscale[iL](features)
            results.append( torch.add( xres, xave ) )
        return results[1:]
