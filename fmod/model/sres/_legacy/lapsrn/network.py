import torch
from torch import Tensor
import torch.nn as nn
import numpy as np
from typing import Any, Mapping, Sequence, Tuple, Union, List, Dict, Literal
import math

Tensors = Sequence[Tensor]
TensorOrTensors = Union[Tensor, Tensors]

def get_upsample_filter(size):
	"""Make a 2D bilinear kernel suitable for upsampling"""
	factor = (size + 1) // 2
	center = (factor - 1) if (size % 2 == 1) else (factor - 0.5)
	og = np.ogrid[:size, :size]
	usfilter = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
	return torch.from_numpy(usfilter).float()


class RecursiveBlock(nn.Module):
    def __init__(self, nlayers: int, nchannels: int = 64):
        super(RecursiveBlock, self).__init__()
        self.block = nn.Sequential()
        for i in range(nlayers):
            self.block.add_module("relu_" + str(i), nn.LeakyReLU(0.2, inplace=True))
            self.block.add_module("conv_" + str(i), nn.Conv2d(in_channels=nchannels, out_channels=nchannels, kernel_size=3, stride=1, padding="same", bias=True))

    def forward(self, x):
        output = self.block(x)
        return output


class FeatureEmbedding(nn.Module):
    def __init__(self, rdepth: int, rlayers: int, nchannels: int):
        super(FeatureEmbedding, self).__init__()
        self.recursive_block = RecursiveBlock(rlayers,nchannels)
        self.num_recursion = rdepth

    def forward(self, x):
        output = x.clone()
        for i in range(self.num_recursion):
            output = self.recursive_block(output) + x
        return output


class LapSrn(nn.Module):
    def __init__(self, nchannels: int, nfeatures: int, rdepth: int, rlayers: int, scale: int ):
        super(LapSrn, self).__init__()
        self.nscale_ops = math.log2(scale)
        self.conv_input = nn.Conv2d(in_channels=nchannels, out_channels=nfeatures, kernel_size=3, stride=1, padding='same', bias=True, )
        self.transpose = nn.ConvTranspose2d(in_channels=nfeatures, out_channels=nfeatures, kernel_size=4, stride=2, padding=1, bias=True)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.scale_img = nn.ConvTranspose2d(in_channels=nchannels, out_channels=nchannels, kernel_size=4, stride=2, padding=1, bias=False)
        self.predict = nn.Conv2d(in_channels=nfeatures, out_channels=nchannels, kernel_size=3, stride=1, padding='same', bias=True)
        self.features = FeatureEmbedding(rdepth, rlayers, nfeatures )
        assert self.nscale_ops.is_integer(), f"Error, scale ({scale}) must be power of 2, math.log2(scale) = {self.nscale_ops}"
        self.init_weights(nfeatures)
        self.use_features = False

    def forward(self, x):
        rescaled_img_2x = self.scale_img(x)
        rescaled_img_4x = self.scale_img(rescaled_img_2x)
        return self.relu( rescaled_img_4x )

    def forward1(self, x):
        features = None
        output_images = []
        rescaled_img = x.clone()

        for i in range(int(self.nscale_ops)):
            rescaled_img = self.scale_img(rescaled_img)
            if self.use_features:
                if features is None:
                    features = self.conv_input(x)
                features = self.features(features)
                features = self.transpose(self.relu(features))
                predict = self.predict(features)
                out =  torch.add(predict, rescaled_img)
            else:
                out = rescaled_img

            out = torch.clamp(out, 0.0, 1.0)
            output_images.append(out)

        return output_images

    def get_targets(self, hr_targ: Tensor ) -> TensorOrTensors:
        targets: List[Tensor] = [ hr_targ ]
        for i in range(int(self.nscale_ops)-1):
            targets.append(  torch.nn.functional.interpolate( targets[-1], scale_factor=0.5, mode='bilinear' ) )
        targets.reverse()
        return targets

    def init_weights( self, nfeatures: int ):
        i_conv = 0
        i_tconv = 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if i_conv == 0: m.weight.data = 0.001 * torch.randn(m.weight.shape)
                else:           m.weight.data = math.sqrt(2 / (3 * 3 * nfeatures)) * torch.randn(m.weight.shape)
                i_conv += 1
                if m.bias is not None: m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                if i_tconv == 0:
                    m.weight.data = math.sqrt(2 / (3 * 3 * nfeatures)) * torch.randn(m.weight.shape)
                else:
                    c1, c2, h, w = m.weight.data.size()
                    weight = get_upsample_filter(h)
                    m.weight.data = weight.view(1, 1, h, w).repeat(c1, c2, 1, 1)
                i_tconv += 1
                if m.bias is not None: m.bias.data.zero_()

def get_model( mconfig: Dict[str, Any] ) -> nn.Module:
    nchannels:          int     = mconfig['nchannels']
    nfeatures:          int     = mconfig['nfeatures']
    downscale_factors: List[int]  = mconfig['downscale_factors']
    unet_depth:         int     = mconfig['unet_depth']
    return LapSrn( nchannels, nfeatures, downscale_factors, unet_depth )
