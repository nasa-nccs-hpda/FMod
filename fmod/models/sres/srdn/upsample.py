import torch, torch.nn as nn
from fmod.models.sres.util import *

class Upsample(nn.Module):

	def __init__(self,
		nchannels_in: int,
		nchannels_out: int,
		scale_factor: int,
		method: str,
		kernel_size: Size2,
		stride: Size2
	):
		super(Upsample, self).__init__()
		if method == "replicate":
			self.usnet = nn.Sequential(
				nn.Conv2d( nchannels_in, nchannels_out, kernel_size, stride=stride, padding='same' ),
				nn.UpsamplingNearest2d( scale_factor=scale_factor )
			)
		elif method == "transpose":
			self.usnet = nn.Sequential(
				nn.ConvTranspose2d( nchannels_in, nchannels_out, kernel_size, stride=scale_factor )
			)
		self.usnet.append( nn.PReLU(init=0.0) )


	def forward(self, x: torch.Tensor) -> torch.Tensor:
		y: torch.Tensor =  self.usnet(x)
		print( f" --- Upsample: {list(x.shape)} -> {list(y.shape)}")
		return  y
