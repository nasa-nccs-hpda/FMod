import torch, torch.nn as nn
from collections import OrderedDict
from fmod.model.sres.common.residual import Residual
from fmod.model.sres.common.upsample import Upsample
from fmod.model.sres.util import *
from fmod.base.util.logging import lgm, exception_handled, log_timing

class SRDN(nn.Module):

	def __init__( self,
		inchannels: int,
		nfeatures: Dict[str,int],
		nrlayers: int,
		scale_factors: List[int],
		usmethod: str,
		kernel_size: Dict[str,int],
		stride: Size2 = 1,
		momentum: float = 0.5
	):
		super(SRDN, self).__init__()

		nchan = nfeatures['hidden']
		ks = kernel_size.get( 'features', kernel_size['hidden'] )
		self.features = nn.Sequential(
			nn.Conv2d( inchannels, nchan, ks, stride, padding="same" ),
			nn.PReLU(init=0.0)
		)

		ks = kernel_size['hidden']
		res_layers = [ ( f"Residual-{iR}", Residual(nchan, ks, stride, momentum) ) for iR in range(nrlayers) ]
		self.residuals = nn.Sequential( OrderedDict( res_layers ) )

		self.global_residual = nn.Sequential(
			nn.Conv2d( nchan, nchan, ks, stride, padding="same" ),
			nn.BatchNorm2d( nchan, momentum=momentum )
		)

		self.upscaling = nn.Sequential()
		nfeatures_in = nchan
		nchan_us = nfeatures['upscale']
		for scale_factor in scale_factors:
			self.upscaling.append( Upsample(nfeatures_in, nchan_us, scale_factor, usmethod, ks, stride ) )
			nfeatures_in = nchan_us

		self.result = nn.Conv2d( nchan_us, nfeatures['output'], kernel_size['output'], stride, padding="same" )

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		f: torch.Tensor = self.features(x)
		r: torch.Tensor = self.residuals( f )
		gr: torch.Tensor = self.global_residual(r)
		y = self.upscaling( f + gr )
		z =  self.result( y )
		lgm().log(f"SRDN.forward: f{list(f.shape)} r{list(r.shape)} gr{list(gr.shape)} y{list(y.shape)} z{list(z.shape)}")
		return z

def get_model( mconfig: Dict[str, Any] ) -> nn.Module:
	nchannels:          int     = mconfig['nchannels']
	nfeatures:   Dict[str,int]  = mconfig['nfeatures']
	scale_factors:   List[int]  = mconfig['downscale_factors']
	kernel_size:  Dict[str,int] = mconfig['kernel_size']
	nrlayers:           int     = mconfig['nrlayers']
	usmethod:           str     = mconfig['usmethod']
	return SRDN( nchannels, nfeatures, nrlayers, scale_factors, usmethod, kernel_size )
