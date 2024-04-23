import torch.nn as nn
import torch
from models.sres.util import *

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
