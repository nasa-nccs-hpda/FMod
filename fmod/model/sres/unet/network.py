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


class MPDownscale(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.maxpool_conv(x)
        return y


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
            self.conv = DoubleConv( in_channels, out_channels )

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

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels, out_channels )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        return self.conv(x)

class UNetUpscale(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv( 2*out_channels, out_channels )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        xup = self.up(x)
        y: torch.Tensor = torch.cat([xup, skip], dim=1 )
        return self.conv(y)


class OutConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, nfeatures: int, depth: int ):
        super(UNet, self).__init__()
        self.depth: int = depth
        self.downscale = nn.ModuleList()
        self.upscale = nn.ModuleList()
        self.base_shape = None

        for iL in range(depth):
            usf, dsf = 2 ** (depth-iL-1), 2 ** iL
            self.downscale.append( MPDownscale(nfeatures * dsf, nfeatures * dsf * 2))
            self.upscale.append( UNetUpscale(nfeatures * usf * 2, nfeatures * usf))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = []
        for iL in range(self.depth):
            skip.insert(0, x)
            x: torch.Tensor = self.downscale[iL](x)
        if self.base_shape is None:
            print( f"\n ----------------- UNet base shape: {list(x.shape)} ----------------- \n")
            self.base_shape = list(x.shape)
        for iL in range(self.depth):
            x = self.upscale[iL](x,skip[iL])
        return x


class UNetSR(nn.Module):
    def __init__(self, n_channels: int, n_features: int, unet_depth: int, n_upscale_ops: int ):
        super(UNetSR, self).__init__()
        self.n_channels: int = n_channels
        self.n_features: int = n_features
        self.workflow = nn.Sequential(
            DoubleConv( n_channels, n_features ),
            UNet( n_features, depth=unet_depth ),
            self.get_upscale_layers( n_upscale_ops ),
            OutConv( n_features, self.n_channels ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.workflow(x)

    def get_upscale_layers(self, nlayers: int) -> nn.Module:
        upscale = nn.Sequential()
        for iL in range(nlayers):
            upscale.add_module( f"ups{iL}", Upscale( self.n_features, self.n_features) )
        return upscale

def get_model( mconfig: Dict[str, Any] ) -> nn.Module:
    nchannels:          int     = mconfig['nchannels']
    nfeatures:          int     = mconfig['nfeatures']
    downscale_factors: List[int]  = mconfig['downscale_factors']
    unet_depth:         int     = mconfig['unet_depth']
    n_upscale_ops = len(downscale_factors)
    return UNetSR( nchannels, nfeatures, unet_depth, n_upscale_ops )