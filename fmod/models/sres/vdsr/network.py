import torch.nn as nn
from typing import Any, List, Tuple, Callable, Optional, Union, overload
from fmod.models.sres.util import *
from fmod.models.sres.common.cnn import BasicBlock

class VDSR(nn.Module):
	def __init__( self,
		conv: Callable[[int,int,Size2,bool],nn.Module],
		nchannels: int,
		nfeatures: int,
		kernel_size: int,
		scale: int,
		n_resblocks: int,
		bn: bool = False,
		act: nn.Module = nn.ReLU(True),
		bias: bool = True,
	):
		super(VDSR, self).__init__()

		def basic_block( in_channels: int, out_channels: int, activation: Optional[nn.Module] ):
			return BasicBlock( conv, in_channels, out_channels, kernel_size, bias=bias, bn=bn, act=activation )

		self.upscaler = nn.Sequential(
			nn.UpsamplingNearest2d(scale_factor=scale),
		)

		m_body = [  basic_block( nchannels, nfeatures, act ) ]
		for _ in range(n_resblocks - 2):
			m_body.append(basic_block( nfeatures, nfeatures, act) )
		m_body.append(basic_block(nfeatures, nchannels, None))

		self.body = nn.Sequential(*m_body)

	def forward(self, x):
		x = self.upscaler(x)
		y = x + self.body(x)
		return y

