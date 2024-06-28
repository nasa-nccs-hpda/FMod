# Deep Back-Projection Networks For Super-Resolution
# https://arxiv.org/abs/1803.02735
from fmod.model.sres.util import *
import torch, math, torch.nn as nn
from fmod.model.sres.common.common import FModule
from .blocks import DenseProjection

def get_model( **config ) -> nn.Module:
	return DBPN(**config)
class DBPN(FModule):

	def __init__( self, **config ):
		parms =  dict( nprojectionfeatures = 32, depth = 2 )
		super(DBPN, self).__init__( parms, **config )

		initial = [
			nn.Conv2d( self.nchannels_in, self.nfeatures, 3, padding=1),
			nn.PReLU(self.nfeatures),
			nn.Conv2d(self.nfeatures, self.nprojectionfeatures, 1),
			nn.PReLU(self.nprojectionfeatures)
		]
		self.initial = nn.Sequential(*initial)

		self.upmodules = nn.ModuleList()
		self.downmodules = nn.ModuleList()
		channels = self.nprojectionfeatures
		for i in range(self.depth):
			self.upmodules.append( DenseProjection(channels, self.nprojectionfeatures, self.scale, True, i > 1) )
			if i != 0:
				channels += self.nprojectionfeatures

		channels = self.nprojectionfeatures
		for i in range(self.depth - 1):
			self.downmodules.append( DenseProjection(channels, self.nprojectionfeatures, self.scale, False, i != 0) )
			channels += self.nprojectionfeatures

		reconstruction = [ nn.Conv2d(self.depth * self.nprojectionfeatures, self.nchannels_out, 3, padding=1) ]
		self.reconstruction = nn.Sequential(*reconstruction)

	def forward(self, x):
		x = self.initial(x)

		h_list = []
		l_list = []
		for i in range(self.depth - 1):
			layer_input = x if i == 0 else torch.cat(l_list, dim=1)
			h_list.append(self.upmodules[i](layer_input))
			l_list.append(self.downmodules[i](torch.cat(h_list, dim=1)))

		h_list.append(self.upmodules[-1](torch.cat(l_list, dim=1)))
		out = self.reconstruction(torch.cat(h_list, dim=1))

		return out
