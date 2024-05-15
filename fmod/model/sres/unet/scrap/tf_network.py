import torch, torch.nn as nn
from collections import OrderedDict
from fmod.models.sres.util import *
import numpy as np


class SRUNET(nn.Module):

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
		super(SRUNET, self).__init__()



	def unet(self, nb_inputs, size_target_domain, shape_inputs, filters):
		# draw the network according to the predictors and target shapes.

		inputs_list = []
		size = np.min([highest_powerof2(shape_inputs[0][0]), highest_powerof2(shape_inputs[0][1])])
		if nb_inputs == 1:
			inputs = Input(shape=shape_inputs[0])
			conv_down = []
			diff_lat = inputs.shape[1] - size + 1
			diff_lon = inputs.shape[2] - size + 1
			conv0 = Conv2D(32, (diff_lat, diff_lon))
			conv0 = BatchNormalization()
			conv0 = Activation('relu')
			prev = conv0
			for i in range(int(log2(size))):
				conv = block_conv(prev, min(filters * int(pow(2, i)), 512))
				pool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv)
				conv_down.append(conv)
				prev = pool
			up = block_conv(prev, filters * min(filters * int(pow(2, i)), 512))
			k = log2(size)
			for i in range(1, int(log2(size_target_domain) + 1)):
				if i <= k:
					up = block_up_conc(up, min(filters * int(pow(2, k - i)), 512), conv_down[int(k - i)])
				else:
					up = block_up(up, filters)
			inputs_list.append(inputs)

		if nb_inputs == 2:
			conv_down = []
			diff_lat = inputs.shape[1] - size + 1
			diff_lon = inputs.shape[2] - size + 1
			conv0 = Conv2D(32, (diff_lat, diff_lon))(inputs)
			conv0 = BatchNormalization()(conv0)
			conv0 = Activation('relu')(conv0)
			prev = conv0
			for i in range(int(log2(size))):
				conv = block_conv(prev, min(filters * int(pow(2, i)), 512))
				pool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv)
				conv_down.append(conv)
				prev = pool

			last_conv = block_conv(prev, min(filters * int(pow(2, i)), 512))
			inputs2 = Input(shape=shape_inputs[1])
			model2 = Dense(filters)(inputs2)
			for i in range(1, int(log2(size))):
				model2 = Dense(min(filters * int(pow(2, i)), 512))(model2)

			merged = concatenate([last_conv, model2])
			up = merged
			k = log2(size)
			for i in range(1, int(log2(size_target_domain) + 1)):
				if i <= k:
					up = block_up_conc(up, min(filters * int(pow(2, k - i)), 512), conv_down[int(k - i)])
				else:
					conv = block_up(up, filters)
					up = conv
			inputs_list.append(inputs)
			inputs_list.append(inputs2)
		last = up
		lastconv = Conv2D(1, 1, padding='same')(last)
		return (Model(inputs=inputs_list, outputs=lastconv))