import torch.fft
from torch.utils.checkpoint import checkpoint
from torch.cuda import amp
import math
from .activations import *
from tltorch.factorized_tensors.core import FactorizedTensor
from fmod.base.util.logging import lgm, exception_handled, log_timing

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
	# Cut & paste from PyTorch official master until it's in a few official releases - RW
	# Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
	def norm_cdf(x):
		# Computes standard normal cumulative distribution function
		return (1. + math.erf(x / math.sqrt(2.))) / 2.

	if (mean < a - 2 * std) or (mean > b + 2 * std):
		lgm().log("mean is more than 2 std from [a, b] in nn.init.trunc_normal_.  The distribution of values may be incorrect.")

	with torch.no_grad():
		# Values are generated by using a truncated uniform distribution and
		# then using the inverse CDF for the normal distribution.
		# Get upper and lower cdf values
		l = norm_cdf((a - mean) / std)
		u = norm_cdf((b - mean) / std)

		# Uniformly fill tensor with values from [l, u], then translate to
		# [2l-1, 2u-1].
		tensor.uniform_(2 * l - 1, 2 * u - 1)

		# Use inverse cdf transform for normal distribution to get truncated
		# standard normal
		tensor.erfinv_()

		# Transform to proper mean, std
		tensor.mul_(std * math.sqrt(2.))
		tensor.add_(mean)

		# Clamp to ensure it's in the proper range
		tensor.clamp_(min=a, max=b)
		return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
	r"""Fills the input Tensor with values drawn from a truncated
	normal distribution. The values are effectively drawn from the
	normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
	with values outside :math:`[a, b]` redrawn until they are within
	the bounds. The method used for generating the random values works
	best when :math:`a \leq \text{mean} \leq b`.
	Args:
	tensor: an n-dimensional `torch.Tensor`
	mean: the mean of the normal distribution
	std: the standard deviation of the normal distribution
	a: the minimum cutoff value
	b: the maximum cutoff value
	Examples:
	>>> w = torch.empty(3, 5)
	>>> nn.init.trunc_normal_(w)
	"""
	return _no_grad_trunc_normal_(tensor, mean, std, a, b)

@torch.jit.script
def drop_path(x: torch.Tensor, drop_prob: float = 0., training: bool = False) -> torch.Tensor:
	"""Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
	This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
	the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
	See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
	changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
	'survival rate' as the argument.
	"""
	if drop_prob == 0. or not training:
		return x
	keep_prob = 1. - drop_prob
	shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2d ConvNets
	random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
	random_tensor.floor_()  # binarize
	output = x.div(keep_prob) * random_tensor
	return output

class DropPath(nn.Module):
	"""Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
	"""

	def __init__(self, drop_prob=None):
		super(DropPath, self).__init__()
		self.drop_prob = drop_prob

	def forward(self, x):
		return drop_path(x, self.drop_prob, self.training)

class MLP(nn.Module):
	def __init__(self,
		in_features,
		hidden_features=None,
		out_features=None,
		act_layer=nn.ReLU,
		output_bias=False,
		drop_rate=0.,
		checkpointing=False,
		gain=1.0):
		super(MLP, self).__init__()
		self.checkpointing = checkpointing
		out_features = out_features or in_features
		hidden_features = hidden_features or in_features

		# Fist dense layer
		fc1 = nn.Conv2d(in_features, hidden_features, 1, bias=True)
		# initialize the weights correctly
		scale = math.sqrt(2.0 / in_features)
		nn.init.normal_(fc1.weight, mean=0., std=scale)
		if fc1.bias is not None:
			nn.init.constant_(fc1.bias, 0.0)

		# activation
		act = act_layer()

		# output layer
		fc2 = nn.Conv2d(hidden_features, out_features, 1, bias=output_bias)
		# gain factor for the output determines the scaling of the output init
		scale = math.sqrt(gain / hidden_features)
		nn.init.normal_(fc2.weight, mean=0., std=scale)
		if fc2.bias is not None:
			nn.init.constant_(fc2.bias, 0.0)

		if drop_rate > 0.:
			drop = nn.Dropout2d(drop_rate)
			self.fwd = nn.Sequential(fc1, act, drop, fc2, drop)
		else:
			self.fwd = nn.Sequential(fc1, act, fc2)

	@torch.jit.ignore
	def checkpoint_forward(self, x):
		return checkpoint(self.fwd, x)

	def forward(self, x):
		if self.checkpointing:
			return self.checkpoint_forward(x)
		else:
			return self.fwd(x)

class RealFFT2(nn.Module):
	"""
	Helper routine to wrap FFT similarly to the SHT
	"""

	def __init__(self,
		nlat,
		nlon,
		lmax=None,
		mmax=None):
		super(RealFFT2, self).__init__()

		self.nlat = nlat
		self.nlon = nlon
		self.lmax = lmax or self.nlat
		self.mmax = mmax or self.nlon // 2 + 1

	def forward(self, x):
		y = torch.fft.rfft2(x, dim=(-2, -1), norm="ortho")
		y = torch.cat((y[..., :math.ceil(self.lmax / 2), :self.mmax], y[..., -math.floor(self.lmax / 2):, :self.mmax]), dim=-2)
		return y

class InverseRealFFT2(nn.Module):
	"""
	Helper routine to wrap FFT similarly to the SHT
	"""

	def __init__(self,
		nlat,
		nlon,
		lmax=None,
		mmax=None):
		super(InverseRealFFT2, self).__init__()

		self.nlat = nlat
		self.nlon = nlon
		self.lmax = lmax or self.nlat
		self.mmax = mmax or self.nlon // 2 + 1

	def forward(self, x):
		return torch.fft.irfft2(x, dim=(-2, -1), s=(self.nlat, self.nlon), norm="ortho")

class SpectralConvS2(nn.Module):
	"""
	Spectral Convolution according to Driscoll & Healy. Designed for convolutions on the two-sphere S2
	using the Spherical Harmonic Transforms in torch-harmonics, but supports convolutions on the periodic
	domain via the RealFFT2 and InverseRealFFT2 wrappers.
	"""

	def __init__(self,
		forward_transform,
		inverse_transform,
		in_channels,
		out_channels,
		gain=2.,
		operator_type="driscoll-healy",
		lr_scale_exponent=0,
		bias=False):
		super(SpectralConvS2, self).__init__()

		self.forward_transform = forward_transform
		self.inverse_transform = inverse_transform

		self.modes_lat = self.inverse_transform.lmax
		self.modes_lon = self.inverse_transform.mmax

		self.scale_residual = (self.forward_transform.nlat != self.inverse_transform.nlat) \
		                      or (self.forward_transform.nlon != self.inverse_transform.nlon)

		# remember factorization details
		self.operator_type = operator_type

		assert self.inverse_transform.lmax == self.modes_lat
		assert self.inverse_transform.mmax == self.modes_lon

		weight_shape = [out_channels, in_channels]

		if self.operator_type == "diagonal":
			weight_shape += [self.modes_lat, self.modes_lon]
			from .contractions import contract_diagonal as _contract
		elif self.operator_type == "block-diagonal":
			weight_shape += [self.modes_lat, self.modes_lon, self.modes_lon]
			from .contractions import contract_blockdiag as _contract
		elif self.operator_type == "driscoll-healy":
			weight_shape += [self.modes_lat]
			from .contractions import contract_dhconv as _contract
		else:
			raise NotImplementedError(f"Unkonw operator type f{self.operator_type}")

		# form weight tensors
		scale = math.sqrt(gain / in_channels) * torch.ones(self.modes_lat, 2)
		scale[0] *= math.sqrt(2)
		self.weight = nn.Parameter(scale * torch.view_as_real(torch.randn(*weight_shape, dtype=torch.complex64)))
		# self.weight = nn.Parameter(scale * torch.randn(*weight_shape, 2))

		# get the right contraction function
		self._contract = _contract

		if bias:
			self.bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1))

	def forward(self, x):
		lgm().log( f'SpectralConvS2.forward: x{tuple(x.shape)}' )
		dtype = x.dtype
		x = x.float()
		residual = x

		with amp.autocast(enabled=False):
			x = self.forward_transform(x)
			if self.scale_residual:
				residual = self.inverse_transform(x)

		x = torch.view_as_real(x)
		x = self._contract(x, self.weight)
		x = torch.view_as_complex(x)

		with amp.autocast(enabled=False):
			x = self.inverse_transform(x)

		if hasattr(self, "bias"):
			x = x + self.bias
		x = x.type(dtype)

		lgm().log(f'SpectralConvS2.result: x{tuple(x.shape)} residual{tuple(residual.shape)}')
		return x, residual

class FactorizedSpectralConvS2(nn.Module):
	"""
	Factorized version of SpectralConvS2. Uses tensorly-torch to keep the weights factorized
	"""

	def __init__(self,
		forward_transform,
		inverse_transform,
		in_channels,
		out_channels,
		gain=2.,
		operator_type="driscoll-healy",
		rank=0.2,
		factorization=None,
		separable=False,
		implementation="factorized",
		decomposition_kwargs=dict(),
		bias=False):
		super(SpectralConvS2, self).__init__()

		self.forward_transform = forward_transform
		self.inverse_transform = inverse_transform

		self.modes_lat = self.inverse_transform.lmax
		self.modes_lon = self.inverse_transform.mmax

		self.scale_residual = (self.forward_transform.nlat != self.inverse_transform.nlat) \
		                      or (self.forward_transform.nlon != self.inverse_transform.nlon)

		# Make sure we are using a Complex Factorized Tensor
		if factorization is None:
			factorization = "Dense"  # No factorization
		if not factorization.lower().startswith("complex"):
			factorization = f"Complex{factorization}"

		# remember factorization details
		self.operator_type = operator_type
		self.rank = rank
		self.factorization = factorization
		self.separable = separable

		assert self.inverse_transform.lmax == self.modes_lat
		assert self.inverse_transform.mmax == self.modes_lon

		weight_shape = [out_channels]

		if not self.separable:
			weight_shape += [in_channels]

		if self.operator_type == "diagonal":
			weight_shape += [self.modes_lat, self.modes_lon]
		elif self.operator_type == "block-diagonal":
			weight_shape += [self.modes_lat, self.modes_lon, self.modes_lon]
		elif self.operator_type == "driscoll-healy":
			weight_shape += [self.modes_lat]
		else:
			raise NotImplementedError(f"Unkonw operator type f{self.operator_type}")

		# form weight tensors
		self.weight = FactorizedTensor.new(weight_shape, rank=self.rank, factorization=factorization,
			fixed_rank_modes=False, **decomposition_kwargs)

		# initialization of weights
		scale = math.sqrt(gain / in_channels)
		self.weight.normal_(0, scale)

		# get the right contraction function
		from .factorizations import get_contract_fun
		self._contract = get_contract_fun(self.weight, implementation=implementation, separable=separable)

		if bias:
			self.bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1))

	def forward(self, x):
		dtype = x.dtype
		x = x.float()
		residual = x

		with amp.autocast(enabled=False):
			x = self.forward_transform(x)
			if self.scale_residual:
				residual = self.inverse_transform(x)

		x = self._contract(x, self.weight, separable=self.separable, operator_type=self.operator_type)

		with amp.autocast(enabled=False):
			x = self.inverse_transform(x)

		if hasattr(self, "bias"):
			x = x + self.bias
		x = x.type(dtype)

		return x, residual
