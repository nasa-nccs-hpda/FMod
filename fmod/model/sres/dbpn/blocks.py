from fmod.model.sres.util import *
import torch, math, torch.nn as nn
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

