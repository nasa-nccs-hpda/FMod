import torch.nn as nn
import torch, math
from typing import Any, List, Tuple, Callable, Optional, Union, overload
from fmod.base.util.config import cfg
from fmod.model.sres.util import *
from fmod.model.sres.common.cnn import BasicBlock, default_conv

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

def get_model( mconfig: Dict[str, Any] ) -> nn.Module:
	nchannels:          int     = mconfig['nchannels']
	nfeatures:          int     = mconfig['nfeatures']
	scale_factors:   List[int]  = mconfig['upscale_factors']
	kernel_size:        Size2   = mconfig['kernel_size']
	nrlayers:           int     = mconfig['nrlayers']
	conv: Callable[[int,int,Size2,bool],nn.Module] = default_conv
	scale: int = math.prod( scale_factors )
	bn: bool = False
	act: nn.Module = nn.ReLU(True)
	bias: bool = True
	return VDSR( conv, nchannels, nfeatures, kernel_size, scale, nrlayers, bn, act, bias )