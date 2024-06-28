from . import blocks
import math, torch.nn as nn
from typing import Any, Dict, List, Tuple, Type, Optional, Union
from fmod.model.sres.common.common import FModule

def get_model( **config ) -> nn.Module:
	return RCAN(**config)
class RCAN(FModule):

	def __init__(self, **kwargs):
		parms = dict(cbottleneck=2,nblocks=20)
		super(RCAN, self).__init__(parms, **kwargs)

		modules_head = [ self.conv(self.nchannels_in, self.nfeatures, self.kernel_size, self.bias) ]
		modules_body = [ ResidualGroup( self.conv, self.nfeatures , self.kernel_size, self.cbottleneck, act=self.act, n_resblocks=self.nblocks) for _ in range(self.nlayers)]
		modules_body.append( self.conv(self.nfeatures , self.nfeatures , self.kernel_size, self.bias))
		modules_tail = [ blocks.Upsampler(self.conv, self.scale, self.nfeatures , act=False), self.conv(self.nfeatures , self.nchannels_out, self.kernel_size, self.bias) ]

		self.head = nn.Sequential(*modules_head)
		self.body = nn.Sequential(*modules_body)
		self.tail = nn.Sequential(*modules_tail)

	def forward(self, x):
		x = self.head(x)
		res = self.body(x)
		res += x
		x = self.tail(res)
		return x


## Channel Attention (CA) Layer
class CALayer(nn.Module):
	def __init__(self, channel, reduction=16):
		super(CALayer, self).__init__()
		# global average pooling: feature --> point
		self.avg_pool = nn.AdaptiveAvgPool2d(1)
		# feature channel downscale and upscale --> channel weight
		self.conv_du = nn.Sequential(
			nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
			nn.ReLU(inplace=True),
			nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
			nn.Sigmoid()
		)

	def forward(self, x):
		y = self.avg_pool(x)
		y = self.conv_du(y)
		return x * y

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
	def __init__( self, conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True)):
		super(RCAB, self).__init__()
		modules_body = []
		for i in range(2):
			modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
			if bn: modules_body.append(nn.BatchNorm2d(n_feat))
			if i == 0: modules_body.append(act)
		modules_body.append(CALayer(n_feat, reduction))
		self.body = nn.Sequential(*modules_body)

	def forward(self, x):
		res = self.body(x)
		res += x
		return res

## Residual Group (RG)
class ResidualGroup(nn.Module):
	def __init__(self, conv, n_feat, kernel_size, reduction, act, n_resblocks):
		super(ResidualGroup, self).__init__()
		modules_body = [ RCAB( conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=act ) for _ in range(n_resblocks) ]
		modules_body.append(conv(n_feat, n_feat, kernel_size))
		self.body = nn.Sequential(*modules_body)

	def forward(self, x):
		res = self.body(x)
		res += x
		return res
