import torch, torch.nn as nn
from collections import OrderedDict
from .residual import Residual
from .upsample import Upsample
from fmod.models.sres.util import *

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
			nn.Conv2d( inchannels, nchan, ks, stride ),
			nn.PReLU(init=0.0)
		)

		ks = kernel_size['hidden']
		res_layers = [ ( f"Residual-{iR}", Residual(nchan, ks, stride, momentum) ) for iR in range(nrlayers) ]
		self.residuals = nn.Sequential( OrderedDict( res_layers ) )

		self.global_residual = nn.Sequential(
			nn.Conv2d( nchan, nchan, ks, stride, padding='same' ),
			nn.BatchNorm2d( nchan, momentum=momentum )
		)

		self.upscaling = nn.Sequential()
		nfeatures_in = nchan
		nchan_us = nfeatures['upscale']
		for scale_factor in scale_factors:
			self.upscaling.append( Upsample(nfeatures_in, nchan_us, scale_factor, usmethod, ks, stride ) )
			nfeatures_in = nchan_us

		self.result = nn.Conv2d( nchan_us, nfeatures['output'], kernel_size['output'], stride, padding='same' )

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		f: torch.Tensor = self.features(x)
		r: torch.Tensor = self.residuals( f )
		gr: torch.Tensor = self.global_residual(r)
		print( f"SRDN.forward: f{list(f.shape)} r{list(r.shape)} gr{list(gr.shape)}" )
		y = f + gr
		y = self.upscaling( y )
		return self.result( y )

# class Generator(object):
#
# 	def __init__(self, noise_shape):
# 		self.noise_shape = noise_shape
#
# 	def generator(self):
# 		init = RandomNormal(stddev=0.02)
#
# 		gen_input = Input(shape=self.noise_shape)
# 		model = Conv2D(filters=64, kernel_size=3, strides=1, padding="same", kernel_initializer=init)(gen_input)
# 		model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1, 2])(model)
#
# 		gen_model = model
#
# 		# Using 16 Residual Blocks
# 		for index in range(16):
# 			model = res_block_gen(model, 3, 64, 1, init)
#
# 		model = Conv2D(filters=64, kernel_size=3, strides=1, padding="same", kernel_initializer=init)(model)
# 		model = BatchNormalization(momentum=0.5)(model)
# 		model = add([gen_model, model])
#
# 		# Using 3 UpSampling Blocks
# 		model = Conv2D(filters=128, kernel_size=3, strides=1, padding="same", kernel_initializer=init)(model)
# 		model = UpSampling2D(size=2)(model)
# 		model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1, 2])(model)
#
# 		model = Conv2D(filters=128, kernel_size=3, strides=1, padding="same", kernel_initializer=init)(model)
# 		model = UpSampling2D(size=3)(model)
# 		model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1, 2])(model)
#
# 		model = Conv2D(filters=128, kernel_size=3, strides=1, padding="same", kernel_initializer=init)(model)
# 		model = UpSampling2D(size=2)(model)
# 		model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1, 2])(model)
#
# 		model = Conv2D(filters=1, kernel_size=9, strides=1, padding="same", kernel_initializer=init)(model)
# 		# model= tfa.image.median_filter2d(model, filter_shape=(37,37))
# 		# model= AveragePooling2D(pool_size=(37,37), strides=(1,1), padding='valid')(model)
# 		# model = Activation("sigmoid")(model)
#
# 		generator_model = Model(inputs=gen_input, outputs=model)
#
# 		return generator_model