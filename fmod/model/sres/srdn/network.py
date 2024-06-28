import torch, torch.nn as nn
from collections import OrderedDict
from fmod.model.sres.common.residual import Residual
from fmod.model.sres.common.upsample import Upsample
from fmod.model.sres.util import *
from fmod.base.util.logging import lgm, exception_handled, log_timing
from fmod.model.sres.common.common import FModule

def get_model( **config ) -> nn.Module:
	return SRDN(**config)
class SRDN(FModule):

	def __init__(self, **kwargs):
		parms = dict(kernel_size={}, stride = 1, momentum = 0.5)
		super(SRDN, self).__init__(parms, **kwargs)

		nfeat = self.nfeatures['hidden']
		self.features = nn.Sequential(
			nn.Conv2d( self.nchannels_in, nfeat, self.kernel_size['features'], self.stride, padding="same" ),
			nn.PReLU(init=0.0)
		)

		ks =  self.kernel_size['hidden']
		res_layers = [ ( f"Residual-{iR}", Residual(nfeat, ks, self.stride, self.momentum) ) for iR in range(self.nlayers) ]
		self.residuals = nn.Sequential( OrderedDict( res_layers ) )

		self.global_residual = nn.Sequential(
			nn.Conv2d( nfeat, nfeat, ks, self.stride, padding="same" ),
			nn.BatchNorm2d( nfeat, momentum=self.momentum )
		)

		self.upscaling = nn.Sequential()
		nfeatures_in = nfeat
		nchan_us = self.nfeatures['upscale']
		for scale_factor in self.downscale_factors:
			self.upscaling.append( Upsample(nfeatures_in, nchan_us, scale_factor, self.usmethod, ks, self.stride ) )
			nfeatures_in = nchan_us

		self.result = nn.Conv2d( nchan_us,  self.nchannels_out, self.kernel_size['output'], self.stride, padding="same" )

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		f: torch.Tensor = self.features(x)
		r: torch.Tensor = self.residuals( f )
		gr: torch.Tensor = self.global_residual(r)
		y = self.upscaling( f + gr )
		z =  self.result( y )
		lgm().log(f"SRDN.forward: f{list(f.shape)} r{list(r.shape)} gr{list(gr.shape)} y{list(y.shape)} z{list(z.shape)}")
		return z

