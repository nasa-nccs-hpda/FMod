import torch.nn as nn
import torch, math
from typing import Any, List, Tuple, Callable, Optional, Union, overload
from fmod.base.util.config import cfg
from fmod.model.sres.util import *
from fmod.model.sres.common.cnn import BasicBlock, default_conv
from fmod.model.sres.common.common import FModule

def get_model( **config ) -> nn.Module:
	return VDSR(**config)

class VDSR(FModule):

	def __init__(self, **kwargs):
		super(VDSR, self).__init__({}, **kwargs)
		self.upscaler = nn.Sequential( nn.UpsamplingNearest2d(scale_factor=self.scale) )
		m_body = [  self.basic_block( self.nchannels_in, self.nfeatures, self.act ) ]
		for _ in range(self.nlayers - 2):
			m_body.append(self.basic_block( self.nfeatures, self.nfeatures, self.act) )
		m_body.append(self.basic_block(self.nfeatures, self.nchannels_out, None))
		self.body = nn.Sequential(*m_body)

	def basic_block(self, in_channels: int, out_channels: int, activation: Optional[nn.Module]):
		return BasicBlock(self.conv, in_channels, out_channels, self.kernel_size, bias=self.bias, bn=self.batch_norm, act=activation)

	def forward(self, x):
		x = self.upscaler(x)
		y = x + self.body(x)
		return y
