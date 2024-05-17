import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Tuple, Type, Optional, Union, Sequence, Mapping
from fmod.base.util.logging import lgm, exception_handled, log_timing
from fmod.model.sres.common.unet import UNet, DoubleConv

class Upscale(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, upscale_fator: int):
        super().__init__()
        self.upscale = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=upscale_fator),
            DoubleConv(out_channels, out_channels )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.upscale(x)

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
    def __init__(self, n_channels: int, nfeatures: int, downscale_factors: List[int], unet_depth: int = 0 ):
        super(MSCNN, self).__init__()
        self.n_channels: int = n_channels
        self.unet_depth: int = unet_depth
        self.downscale_factors = downscale_factors
        self.inc: nn.Module = DoubleConv( n_channels, nfeatures )
        self.upscale: nn.ModuleList = nn.ModuleList()
        self.upsample: nn.ModuleList = nn.ModuleList()
        self.crossscale: nn.ModuleList = nn.ModuleList()
        self.unet: Optional[UNet] = UNet( nfeatures, unet_depth ) if unet_depth > 0 else None
        for iL, usf in enumerate(downscale_factors):
            self.upscale.append(  Upscale( nfeatures, nfeatures, usf ) )
            self.crossscale.append(  Crossscale( nfeatures, self.n_channels ) )
            self.upsample.append( Upsample(usf) )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features, results = self.inc(x), [x]
        if self.unet_depth > 0:
            features = self.unet(features)
        for iL, usf in enumerate(self.downscale_factors):
            features = self.upscale[iL](features)
            xave = self.upsample[iL](results[-1])
            xres = self.crossscale[iL](features)
            results.append( torch.add( xres, xave ) )
        return results[1:]

def get_model( mconfig: Dict[str, Any] ) -> nn.Module:
    nchannels:          int     = mconfig['nchannels']
    nfeatures:          int     = mconfig['nfeatures']
    downscale_factors: List[int]  = mconfig['downscale_factors']
    unet_depth:         int     = mconfig['unet_depth']
    return MSCNN( nchannels, nfeatures, downscale_factors, unet_depth )
