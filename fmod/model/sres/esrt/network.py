from . import blocks
import torch, math
import torch.nn as nn
import torch.nn.functional as F
from fmod.model.sres.common.tools import extract_image_patches, reduce_mean, reduce_sum, same_padding, reverse_patches
from fmod.model.sres.common.transformer import drop_path, DropPath, PatchEmbed, Mlp, MLABlock
from fmod.model.sres.common.position import PositionEmbeddingLearned, PositionEmbeddingSine
from typing import Any, Dict, List, Tuple, Type, Optional, Union, Sequence, Mapping, Callable
from fmod.model.sres.common.common import FModule

def get_model( **config ) -> nn.Module:
	return ESRT(**config)
class ESRT(FModule):

	def __init__(self, **kwargs):
		super(ESRT, self).__init__({}, **kwargs)
		modules_head = [ self.conv(self.nchannels_in, self.nfeatures, self.kernel_size, self.bias) ]
		modules_body = nn.ModuleList()
		for i in range(self.nlayers):
			modules_body.append( Un(n_feats=self.nfeatures, wn=self.wn) )

		modules_tail = [
			blocks.Upsampler(self.conv, self.scale, self.nfeatures, act=False),
			self.conv(self.nfeatures, self.nchannels_out, self.kernel_size) ]

		self.up = nn.Sequential(  blocks.Upsampler(self.conv, self.scale, self.nfeatures, act=False),
								  BasicConv(self.nfeatures, self.nchannels_out, 3, 1, 1) )
		self.head = nn.Sequential(*modules_head)
		self.body = nn.Sequential(*modules_body)
		self.tail = nn.Sequential(*modules_tail)
		self.reduce = self.conv( self.nlayers*self.nfeatures,  self.nfeatures,  self.kernel_size )

	def forward(self, x1, x2=None, test=False):
		x1 = self.head(x1)
		res2 = x1
		body_out = [ self.body[i](x1) for i in range(self.n_blocks) ]
		res1 = torch.cat(body_out, 1)
		res1 = self.reduce(res1)
		x1 = self.tail(res1)
		x1 = self.up(res2) + x1
		return x1


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

class one_conv(nn.Module):
	def __init__(self, inchanels, growth_rate, kernel_size=3, relu=True):
		super(one_conv, self).__init__()
		wn = lambda x: torch.nn.utils.weight_norm(x)
		self.conv = nn.Conv2d(inchanels, growth_rate, kernel_size=kernel_size, padding=kernel_size >> 1, stride=1)
		self.flag = relu
		self.conv1 = nn.Conv2d(growth_rate, inchanels, kernel_size=kernel_size, padding=kernel_size >> 1, stride=1)
		if relu:
			self.relu = nn.PReLU(growth_rate)
		self.weight1 = blocks.Scale(1)
		self.weight2 = blocks.Scale(1)

	def forward(self, x):
		if not self.flag:   output = self.weight1(x) + self.weight2(self.conv1(self.conv(x)))
		else:               output = self.weight1(x) + self.weight2(self.conv1(self.relu(self.conv(x))))
		return output  # torch.cat((x,output),1)

class BasicConv(nn.Module):
	def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=1, dilation=1, groups=1, relu=True,
		bn=False, bias=False, up_size=0, fan=False):
		super(BasicConv, self).__init__()
		wn = lambda x: torch.nn.utils.weight_norm(x)
		self.out_channels = out_planes
		self.in_channels = in_planes
		if fan:
			self.conv = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
				dilation=dilation, groups=groups, bias=bias)
		else:
			self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
				dilation=dilation, groups=groups, bias=bias)
		self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
		self.relu = nn.ReLU(inplace=True) if relu else None
		self.up_size = up_size
		self.up_sample = nn.Upsample(size=(up_size, up_size), mode='bilinear') if up_size != 0 else None

	def forward(self, x):
		x = self.conv(x)
		if self.bn is not None:
			x = self.bn(x)
		if self.relu is not None:
			x = self.relu(x)
		if self.up_size > 0:
			x = self.up_sample(x)
		return x

class one_module(nn.Module):
	def __init__(self, n_feats):
		super(one_module, self).__init__()
		self.layer1 = one_conv(n_feats, n_feats // 2, 3)
		self.layer2 = one_conv(n_feats, n_feats // 2, 3)
		# self.layer3 = one_conv(n_feats, n_feats//2,3)
		self.layer4 = BasicConv(n_feats, n_feats, 3, 1, 1)
		self.alise = BasicConv(2 * n_feats, n_feats, 1, 1, 0)
		self.atten = CALayer(n_feats)
		self.weight1 = blocks.Scale(1)
		self.weight2 = blocks.Scale(1)
		self.weight3 = blocks.Scale(1)
		self.weight4 = blocks.Scale(1)
		self.weight5 = blocks.Scale(1)

	def forward(self, x):
		x1 = self.layer1(x)
		x2 = self.layer2(x1)
		# x3 = self.layer3(x2)
		x4 = self.layer4(self.atten(self.alise(torch.cat([self.weight2(x2), self.weight3(x1)], 1))))
		return self.weight4(x) + self.weight5(x4)

class Updownblock(nn.Module):
	def __init__(self, n_feats):
		super(Updownblock, self).__init__()
		self.encoder = one_module(n_feats)
		self.decoder_low = one_module(n_feats)  # nn.Sequential(one_module(n_feats),
		#                     one_module(n_feats),
		#                     one_module(n_feats))
		self.decoder_high = one_module(n_feats)
		self.alise = one_module(n_feats)
		self.alise2 = BasicConv(2 * n_feats, n_feats, 1, 1, 0)  # one_module(n_feats)
		self.down = nn.AvgPool2d(kernel_size=2)
		self.att = CALayer(n_feats)

	def forward(self, x):
		x1 = self.encoder(x)
		x2 = self.down(x1)
		high = x1 - F.interpolate(x2, size=x.size()[-2:], mode='bilinear', align_corners=True)
		for i in range(5):
			x2 = self.decoder_low(x2)
		x3 = x2
		# x3 = self.decoder_low(x2)
		high1 = self.decoder_high(high)
		x4 = F.interpolate(x3, size=x.size()[-2:], mode='bilinear', align_corners=True)
		return self.alise(self.att(self.alise2(torch.cat([x4, high1], dim=1)))) + x

class Un(nn.Module):
	def __init__(self, n_feats, wn):
		super(Un, self).__init__()
		self.encoder1 = Updownblock(n_feats)
		self.encoder2 = Updownblock(n_feats)
		self.encoder3 = Updownblock(n_feats)
		self.reduce = blocks.default_conv(3 * n_feats, n_feats, 3)
		self.weight2 = blocks.Scale(1)
		self.weight1 = blocks.Scale(1)
		self.attention = MLABlock(n_feat=n_feats, dim=288)
		self.alise = blocks.default_conv(n_feats, n_feats, 3)

	def forward(self, x):
		# out = self.encoder3(self.encoder2(self.encoder1(x)))
		x1 = self.encoder1(x)
		x2 = self.encoder2(x1)
		x3 = self.encoder3(x2)
		out = x3
		b, c, h, w = x3.shape
		out = self.attention(self.reduce(torch.cat([x1, x2, x3], dim=1)))
		out = out.permute(0, 2, 1)
		out = reverse_patches(out, (h, w), (3, 3), 1, 1)
		out = self.alise(out)

		return self.weight1(x) + self.weight2(out)


