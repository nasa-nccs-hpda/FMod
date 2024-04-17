import torch.nn as nn
from .util import *

class Upsample(nn.Module):

	def __init__(self, nchannels_in: int, nchannels_out: int, scale_factor: int, kernel_size: Size2, stride: Size2 ):
		super(Upsample, self).__init__()
		self.usnet = nn.Sequential(
			nn.Conv2d( nchannels_in, nchannels_out, kernel_size, stride=stride, padding='same' ),
			nn.UpsamplingNearest2d( scale_factor=scale_factor ),
			nn.PReLU( init=0.0 ),
		)


	def forward(self, x):
		return self.usnet(x)
