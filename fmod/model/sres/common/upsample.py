import torch, math, torch.nn as nn
from fmod.model.sres.util import *
from fmod.base.util.logging import lgm, exception_handled, log_timing

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
		print( f"Upsample: x{list(x.shape)}")
		y: torch.Tensor =  self.usnet(x)
		return  y

class SPUpsample(nn.Sequential):
	# Sub-Pixel Upsampling
	def __init__(self,
		conv: Callable[[int,int,Size2,bool],nn.Module],
		scale: int,
		nchannels: int,
		bn: bool = False,
		act: Optional[str] = False,
		bias: bool = True
	):
		m = []
		if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
			for _ in range(int(math.log(scale, 2))):
				m.append(conv(nchannels, 4 * nchannels, 3, bias))
				m.append(nn.PixelShuffle(2))
				if bn:
					m.append(nn.BatchNorm2d(nchannels))
				if act == 'relu':
					m.append(nn.ReLU(True))
				elif act == 'prelu':
					m.append(nn.PReLU(nchannels))

		elif scale == 3:
			m.append(conv(nchannels, 9 * nchannels, 3, bias))
			m.append(nn.PixelShuffle(3))
			if bn:
				m.append(nn.BatchNorm2d(nchannels))
			if act == 'relu':
				m.append(nn.ReLU(True))
			elif act == 'prelu':
				m.append(nn.PReLU(nchannels))
		else:
			raise NotImplementedError

		super(SPUpsample, self).__init__(*m)


