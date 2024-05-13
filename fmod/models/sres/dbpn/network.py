# Deep Back-Projection Networks For Super-Resolution
# https://arxiv.org/abs/1803.02735

from fmod.models.sres.util import *
import torch, math, torch.nn as nn
from fmod.models.sres.common.cnn import default_conv

conv_spec = { 2: (6, 2, 2),  4: (8, 4, 2),  8: (12, 8, 2) }

def projection_conv( in_channels: int, out_channels: int, scale: int, upscale=True ):
	kernel_size, stride, padding = conv_spec[scale]
	conv_f = nn.ConvTranspose2d if upscale else nn.Conv2d
	return conv_f( in_channels, out_channels, kernel_size, stride=stride, padding=padding )

class DenseProjection(nn.Module):
	def __init__(self,
		in_channels: int,
		nfeatures: int,
		scale: int,
		upscale: bool =True,
		bottleneck: bool =True
	):
		super(DenseProjection, self).__init__()
		if bottleneck:
			self.bottleneck = nn.Sequential(*[
				nn.Conv2d(in_channels, nfeatures, 1),
				nn.PReLU(nfeatures)
			])
			inter_channels = nfeatures
		else:
			self.bottleneck = None
			inter_channels = in_channels

		self.conv_1 = nn.Sequential(*[
			projection_conv(inter_channels, nfeatures, scale, upscale),
			nn.PReLU(nfeatures)
		])
		self.conv_2 = nn.Sequential(*[
			projection_conv(nfeatures, inter_channels, scale, not upscale),
			nn.PReLU(inter_channels)
		])
		self.conv_3 = nn.Sequential(*[
			projection_conv(inter_channels, nfeatures, scale, upscale),
			nn.PReLU(nfeatures)
		])

	def forward(self, x):
		if self.bottleneck is not None:
			x = self.bottleneck(x)

		a_0 = self.conv_1(x)
		b_0 = self.conv_2(a_0)
		e = b_0.sub(x)
		a_1 = self.conv_3(e)

		out = a_0.add(a_1)

		return out

class DBPN(nn.Module):
	def __init__( self,
        nchannels: int,
		scale: int = 2,
		nfeatures: int = 128,
		nprojectionfeatures: int = 32,
		depth: int = 2
	):
		super(DBPN, self).__init__()
		self.depth = depth

		initial = [
			nn.Conv2d( nchannels, nfeatures, 3, padding=1),
			nn.PReLU(nfeatures),
			nn.Conv2d(nfeatures, nprojectionfeatures, 1),
			nn.PReLU(nprojectionfeatures)
		]
		self.initial = nn.Sequential(*initial)

		self.upmodules = nn.ModuleList()
		self.downmodules = nn.ModuleList()
		channels = nprojectionfeatures
		for i in range(self.depth):
			self.upmodules.append( DenseProjection(channels, nprojectionfeatures, scale, True, i > 1) )
			if i != 0:
				channels += nprojectionfeatures

		channels = nprojectionfeatures
		for i in range(self.depth - 1):
			self.downmodules.append(
				DenseProjection(channels, nprojectionfeatures, scale, False, i != 0)
			)
			channels += nprojectionfeatures

		reconstruction = [
			nn.Conv2d(self.depth * nprojectionfeatures, nchannels, 3, padding=1)
		]
		self.reconstruction = nn.Sequential(*reconstruction)

	def forward(self, x):
		x = self.initial(x)

		h_list = []
		l_list = []
		for i in range(self.depth - 1):
			layer_input = x if i == 0 else torch.cat(l_list, dim=1)
			h_list.append(self.upmodules[i](layer_input))
			l_list.append(self.downmodules[i](torch.cat(h_list, dim=1)))

		h_list.append(self.upmodules[-1](torch.cat(l_list, dim=1)))
		out = self.reconstruction(torch.cat(h_list, dim=1))

		return out

def get_model( mconfig: Dict[str, Any] ) -> nn.Module:
	nchannels:          int     = mconfig['nchannels']
	nfeatures:          int     = mconfig['nfeatures']
	depth:              int     = mconfig['depth']
	nprojectionfeatures: int    = mconfig['nprojectionfeatures']
	scale_factors:   List[int]  = mconfig['upscale_factors']
	scale:              int = math.prod(scale_factors)
	return DBPN(nchannels, scale, nfeatures, nprojectionfeatures, depth)