from . import blocks
import torch, math
import torch.nn as nn
from typing import Any, Dict, List, Tuple, Type, Optional, Union, Sequence, Mapping
from fmod.model.sres.common.common import FModule

def get_model( **config ) -> nn.Module:
	return LatticeNet(**config)
class LatticeNet(FModule):

	def __init__(self, **kwargs):
		parms = dict(ndiff=16)
		super(LatticeNet, self).__init__(parms, **kwargs)

		self.conv1 = nn.Conv2d(self.nchannels_in, self.nfeatures, kernel_size=self.kernel_size, padding=self.kernel_size//2, bias=self.bias)
		self.conv2 = nn.Conv2d(self.nfeatures, self.nfeatures, kernel_size=self.kernel_size, padding=self.kernel_size//2, bias=self.bias)

		# define body module
		self.body_unit1 = LatticeBlock(self.nfeatures , self.ndiff)
		self.body_unit2 = LatticeBlock(self.nfeatures , self.ndiff)
		self.body_unit3 = LatticeBlock(self.nfeatures , self.ndiff)
		self.body_unit4 = LatticeBlock(self.nfeatures , self.ndiff)

		self.T_tdm1 = nn.Sequential(
			nn.Conv2d(self.nfeatures , self.nfeatures  // 2, kernel_size=1, padding=0, bias=self.bias),
			nn.ReLU())
		self.L_tdm1 = nn.Sequential(
			nn.Conv2d(self.nfeatures , self.nfeatures  // 2, kernel_size=1, padding=0, bias=self.bias),
			nn.ReLU())

		self.T_tdm2 = nn.Sequential(
			nn.Conv2d(self.nfeatures , self.nfeatures  // 2, kernel_size=1, padding=0, bias=self.bias),
			nn.ReLU())
		self.L_tdm2 = nn.Sequential(
			nn.Conv2d(self.nfeatures , self.nfeatures  // 2, kernel_size=1, padding=0, bias=self.bias),
			nn.ReLU())

		self.T_tdm3 = nn.Sequential(
			nn.Conv2d(self.nfeatures , self.nfeatures  // 2, kernel_size=1, padding=0, bias=self.bias),
			nn.ReLU())
		self.L_tdm3 = nn.Sequential(
			nn.Conv2d(self.nfeatures , self.nfeatures  // 2, kernel_size=1, padding=0, bias=self.bias),
			nn.ReLU())

		modules_tail = [nn.Conv2d(self.nfeatures , self.nfeatures , kernel_size=3, padding=1, bias=self.bias),
			nn.Conv2d(self.nfeatures , self.nchannels_out * (self.scale ** 2), kernel_size=3, padding=1, bias=self.bias),
			nn.PixelShuffle(self.scale)]
		self.tail = nn.Sequential(*modules_tail)

	def forward(self, x):
		x = self.conv1(x)
		x = self.conv2(x)

		res1 = self.body_unit1(x)
		res2 = self.body_unit2(res1)
		res3 = self.body_unit3(res2)
		res4 = self.body_unit4(res3)

		T_tdm1 = self.T_tdm1(res4)
		L_tdm1 = self.L_tdm1(res3)
		out_TDM1 = torch.cat((T_tdm1, L_tdm1), 1)

		T_tdm2 = self.T_tdm2(out_TDM1)
		L_tdm2 = self.L_tdm2(res2)
		out_TDM2 = torch.cat((T_tdm2, L_tdm2), 1)

		T_tdm3 = self.T_tdm3(out_TDM2)
		L_tdm3 = self.L_tdm3(res1)
		out_TDM3 = torch.cat((T_tdm3, L_tdm3), 1)

		res = out_TDM3 + x
		out = self.tail(res)
		return out

class CC(nn.Module):
	def __init__(self, channel, reduction=16):
		super(CC, self).__init__()
		self.avg_pool = nn.AdaptiveAvgPool2d(1)
		self.conv_mean = nn.Sequential(
			nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
			nn.ReLU(inplace=True),
			nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
			nn.Sigmoid()
		)
		self.conv_std = nn.Sequential(
			nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
			nn.ReLU(inplace=True),
			nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
			nn.Sigmoid()
		)

	def forward(self, x):
		ca_mean = self.avg_pool(x)
		ca_mean = self.conv_mean(ca_mean)

		# std
		m_batchsize, C, height, width = x.size()
		x_dense = x.view(m_batchsize, C, -1)
		ca_std = torch.std(x_dense, dim=2, keepdim=True)
		ca_std = ca_std.view(m_batchsize, C, 1, 1)
		ca_var = self.conv_std(ca_std)

		# Coefficient of Variation
		# # cv1 = ca_std / ca_mean
		# cv = torch.div(ca_std, ca_mean)
		# ram = self.sigmoid(ca_mean + ca_var)

		cc = (ca_mean + ca_var) / 2.0
		return cc

class LatticeBlock(nn.Module):
	def __init__(self, nFeat, nDiff, nFeat_slice=4):
		super(LatticeBlock, self).__init__()

		self.D3 = nFeat
		self.d = nDiff
		self.s = nFeat_slice

		blocks = [
			nn.Conv2d(nFeat, nFeat - nDiff, kernel_size=3, padding=1, bias=True),
			nn.LeakyReLU(0.05),
			nn.Conv2d(nFeat - nDiff, nFeat - nDiff, kernel_size=3, padding=1, bias=True),
			nn.LeakyReLU(0.05),
			nn.Conv2d(nFeat - nDiff, nFeat, kernel_size=3, padding=1, bias=True),
			nn.LeakyReLU(0.05)
		]
		self.conv_block0 = nn.Sequential(*blocks)

		self.fea_ca1 = CC(nFeat)
		self.x_ca1 = CC(nFeat)

		blocks = [
			nn.Conv2d(nFeat, nFeat - nDiff, kernel_size=3, padding=1, bias=True),
			nn.LeakyReLU(0.05),
			nn.Conv2d(nFeat - nDiff, nFeat - nDiff, kernel_size=3, padding=1, bias=True),
			nn.LeakyReLU(0.05),
			nn.Conv2d(nFeat - nDiff, nFeat, kernel_size=3, padding=1, bias=True),
			nn.LeakyReLU(0.05)
		]
		self.conv_block1 = nn.Sequential(*blocks)

		self.fea_ca2 = CC(nFeat)
		self.x_ca2 = CC(nFeat)

		self.compress = nn.Conv2d(2 * nFeat, nFeat, kernel_size=1, padding=0, bias=True)

	def forward(self, x):
		# analyse unit
		x_feature_shot = self.conv_block0(x)
		fea_ca1 = self.fea_ca1(x_feature_shot)
		x_ca1 = self.x_ca1(x)

		p1z = x + fea_ca1 * x_feature_shot
		q1z = x_feature_shot + x_ca1 * x

		# synthes_unit
		x_feat_long = self.conv_block1(p1z)
		fea_ca2 = self.fea_ca2(q1z)
		p3z = x_feat_long + fea_ca2 * q1z
		x_ca2 = self.x_ca2(x_feat_long)
		q3z = q1z + x_ca2 * x_feat_long

		out = torch.cat((p3z, q3z), 1)
		out = self.compress(out)

		return out
