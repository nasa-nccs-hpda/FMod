import torch.nn as nn
import torch
from fmod.models.sres.util import *

class Residual(nn.Module):

	def __init__(self,
		nchannels: int,
		kernel_size: Size2,
		stride: Size2,
		momentum: float = 0.5
	):
		super(Residual, self).__init__()
		self.rnet = nn.Sequential(
			nn.Conv2d( nchannels, nchannels, kernel_size, stride=stride, padding='same' ),
			nn.BatchNorm2d( nchannels, momentum=momentum ),
			nn.PReLU( init=0.0 ),
			nn.Conv2d( nchannels, nchannels, kernel_size, stride=stride, padding='same' ),
			nn.BatchNorm2d( nchannels, momentum=momentum )
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return x + self.rnet(x)


class ResBlock(nn.Module):
	def __init__( self,
		conv: Callable[[int,int,Size2,bool],nn.Module],
		nchannels: int,
		kernel_size: Size2,
		bias: bool = True,
		bn: bool = False,
		act: nn.Module = nn.ReLU(True),
		res_scale: float = 1
	):
		super(ResBlock, self).__init__()
		m = []
		for i in range(2):
			m.append(conv(nchannels, nchannels, kernel_size, bias))
			if bn: m.append(nn.BatchNorm2d(nchannels))
			if i == 0: m.append(act)

		self.body = nn.Sequential(*m)
		self.res_scale = res_scale

	def forward(self, x):
		res = self.body(x).mul(self.res_scale)
		res += x

		return res
