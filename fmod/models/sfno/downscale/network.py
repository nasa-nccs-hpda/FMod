from fmod.base.util.config import cfg
from fmod.models.sfno.sht import InverseRealSHT, RealSHT
from ..layers import *
from functools import partial
from fmod.base.util.logging import lgm, exception_handled, log_timing
from fmod.pipeline.trainer import TaskType
from fmod.base.util.ops import nnan, pctnan, pctnant

class SpectralFilterLayer(nn.Module):
	"""
	Fourier layer. Contains the convolution part of the FNO/SFNO
	"""

	def __init__(
		self,
		forward_transform,
		inverse_transform,
		input_dim,
		output_dim,
		gain=2.,
		operator_type="diagonal",
		hidden_size_factor=2,
		factorization=None,
		separable=False,
		rank=1e-2,
		bias=True):
		super(SpectralFilterLayer, self).__init__()

		if factorization is None:
			self.filter = SpectralConvS2(forward_transform,
				inverse_transform,
				input_dim,
				output_dim,
				gain=gain,
				operator_type=operator_type,
				bias=bias)

		elif factorization is not None:
			self.filter = FactorizedSpectralConvS2(forward_transform,
				inverse_transform,
				input_dim,
				output_dim,
				gain=gain,
				operator_type=operator_type,
				rank=rank,
				factorization=factorization,
				separable=separable,
				bias=bias)

		else:
			raise (NotImplementedError)

	def forward(self, x):
		return self.filter(x)

class SphericalFourierNeuralOperatorBlock(nn.Module):
	"""
	Helper module for a single SFNO/FNO block. Can use both FFTs and SHTs to represent either FNO or SFNO blocks.
	"""

	def __init__(
		self,
		forward_transform,
		inverse_transform,
		input_dim,
		output_dim,
		operator_type="driscoll-healy",
		mlp_ratio=2.,
		drop_rate=0.,
		drop_path=0.,
		act_layer=nn.ReLU,
		norm_layer=nn.Identity,
		factorization=None,
		separable=False,
		rank=128,
		inner_skip="linear",
		outer_skip=None,
		use_mlp=True):
		super(SphericalFourierNeuralOperatorBlock, self).__init__()

		if act_layer == nn.Identity:
			gain_factor = 1.0
		else:
			gain_factor = 2.0

		if inner_skip == "linear" or inner_skip == "identity":
			gain_factor /= 2.0

		# convolution layer
		self.filter = SpectralFilterLayer(
			forward_transform,
			inverse_transform,
			input_dim,
			output_dim,
			gain=gain_factor,
			operator_type=operator_type,
			hidden_size_factor=mlp_ratio,
			factorization=factorization,
			separable=separable,
			rank=rank,
			bias=True)

		if inner_skip == "linear":
			self.inner_skip = nn.Conv2d(input_dim, output_dim, 1, 1)
			nn.init.normal_(self.inner_skip.weight, std=math.sqrt(gain_factor / input_dim))
		elif inner_skip == "identity":
			assert input_dim == output_dim
			self.inner_skip = nn.Identity()
		elif inner_skip == "none":
			pass
		else:
			raise ValueError(f"Unknown skip connection type {inner_skip}")

		self.act_layer = act_layer()

		# first normalisation layer
		self.norm0 = norm_layer()

		# dropout
		self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

		gain_factor = 1.0
		if outer_skip == "linear" or inner_skip == "identity":
			gain_factor /= 2.

		if use_mlp:
			mlp_hidden_dim = int(output_dim * mlp_ratio)
			self.mlp = MLP(in_features=output_dim,
				out_features=input_dim,
				hidden_features=mlp_hidden_dim,
				act_layer=act_layer,
				drop_rate=drop_rate,
				checkpointing=False,
				gain=gain_factor)

		if outer_skip == "linear":
			self.outer_skip = nn.Conv2d(input_dim, input_dim, 1, 1)
			torch.nn.init.normal_(self.outer_skip.weight, std=math.sqrt(gain_factor / input_dim))
		elif outer_skip == "identity":
			assert input_dim == output_dim
			self.outer_skip = nn.Identity()
		elif outer_skip == "none":
			pass
		else:
			raise ValueError(f"Unknown skip connection type {outer_skip}")

		# second normalisation layer
		self.norm1 = norm_layer()

	# def init_weights(self, scale):
	#     if hasattr(self, "inner_skip") and isinstance(self.inner_skip, nn.Conv2d):
	#         gain_factor = 1.
	#         scale = (gain_factor / embed_chans)**0.5
	#         nn.init.normal_(self.inner_skip.weight, mean=0., std=scale)
	#         self.filter.filter.init_weights(scale)
	#     else:
	#         gain_factor = 2.
	#         scale = (gain_factor / embed_chans)**0.5
	#         self.filter.filter.init_weights(scale)

	def forward(self, x):
		lgm().log( f" *** SphericalFourierNeuralOperatorBlock.forward: x{tuple(x.shape)}")
		x1, residual = self.filter(x)
		lgm().log(f" *** filter: residual{tuple(residual.shape)}, x{tuple(x.shape)} -> y{tuple(x1.shape)}")

		x = self.norm0(x1)

		if hasattr(self, "inner_skip"):
			isr = self.inner_skip(residual)
			lgm().log(f" *** inner_skip: residual{tuple(residual.shape)}, isr{tuple(isr.shape)}, x{tuple(x.shape)}")
			x = x + isr

		if hasattr(self, "act_layer"):
			x = self.act_layer(x)

		if hasattr(self, "mlp"):
			xsh0 = tuple(x.shape)
			x = self.mlp(x)
			lgm().log(f" *** mlp: input{xsh0}, output{tuple(x.shape)}")

		x = self.norm1(x)

		x = self.drop_path(x)

		if hasattr(self, "outer_skip"):
			osr = self.outer_skip(residual)
			lgm().log(f" *** outer_skip: residual{tuple(residual.shape)}, isr{tuple(osr.shape)}, x{tuple(x.shape)}")
			x = x + osr

		lgm().log(f" *** SphericalFourierNeuralOperatorBlock.result: x{tuple(x.shape)}")
		return x

sfno_network_parms = [ 'spectral_transform','operator_type', 'in_chans', 'out_chans', 'pos_embed', 'normalization_layer', 'spectral_transform', 'operator_type'
		        'scale_factor', 'embed_chans', 'embed_chans', 'num_layers', 'encoder_layers', 'mlp_ratio', 'drop_rate', 'drop_path_rate', 'hard_thresholding_fraction',
		        'big_skip', 'factorization', 'separable', 'rank', 'activation_function', 'use_mlp' ]

class SphericalFourierNeuralOperatorNet(nn.Module):


	"""
	SphericalFourierNeuralOperator module. Can use both FFTs and SHTs to represent either FNO or SFNO,
	both linear and non-linear variants.

	Parameters
	----------
	spectral_transform : str, optional
		Type of spectral transformation to use, by default "sht"
	operator_type : str, optional
		Type of operator to use ('driscoll-healy', 'diagonal'), by default "driscoll-healy"
	in_chans : int, optional
		Number of input channels, by default 3
	out_chans : int, optional
		Number of output channels, by default 3
	embed_chans : int, optional
		Dimension of the embeddings, by default 256
	num_layers : int, optional
		Number of layers in the network, by default 4
	activation_function : str, optional
		Activation function to use, by default "gelu"
	encoder_layers : int, optional
		Number of layers in the encoder, by default 1
	use_mlp : int, optional
		Whether to use MLPs in the SFNO blocks, by default True
	mlp_ratio : int, optional
		Ratio of MLP to use, by default 2.0
	drop_rate : float, optional
		Dropout rate, by default 0.0
	drop_path_rate : float, optional
		Dropout path rate, by default 0.0
	normalization_layer : str, optional
		Type of normalization layer to use ("layer_norm", "instance_norm", "none"), by default "instance_norm"
	hard_thresholding_fraction : float, optional
		Fraction of hard thresholding (frequency cutoff) to apply, by default 1.0
	big_skip : bool, optional
		Whether to add a single large skip connection, by default True
	rank : float, optional
		Rank of the approximation, by default 1.0
	factorization : Any, optional
		Type of factorization to use, by default None
	separable : bool, optional
		Whether to use separable convolutions, by default False
	rank : (int, Tuple[int]), optional
		If a factorization is used, which rank to use. Argument is passed to tensorly
	pos_embed : bool, optional
		Whether to use positional embedding, by default True

	Example:
	--------
	>>> model = SphericalFourierNeuralOperatorNet(
	...         img_shape=(128, 256),
	...         scale_factor=4,
	...         in_chans=2,
	...         out_chans=2,
	...         embed_chans=16,
	...         num_layers=4,
	...         use_mlp=True,)
	>>> model(torch.randn(1, 2, 128, 256)).shape
	torch.Size([1, 2, 128, 256])
	"""

	def __init__(
		self,
		spectral_transform="sht",
		operator_type="driscoll-healy",
		in_shape=(128, 256),
		out_shape=(128, 256),
		grid="equiangular",
		in_chans=3,
		out_chans=3,
		embed_chans=256,
		num_layers=4,
		activation_function="relu",
		encoder_layers=1,
		use_mlp=True,
		mlp_ratio=1.,
		drop_rate=0.,
		drop_path_rate=0.,
		normalization_layer="none",
		hard_thresholding_fraction=1.0,
		big_skip=False,
		factorization=None,
		separable=False,
		rank=128,
		pos_embed=False):
		super(SphericalFourierNeuralOperatorNet, self).__init__()

		lgm().log("SphericalFourierNeuralOperatorNet")
		lgm().log(f" -> spectral_transform = {spectral_transform}")
		lgm().log(f" -> operator_type = {operator_type}")
		lgm().log(f" -> grid = {grid}")
		lgm().log(f" -> in_shape = {list(in_shape)}")
		lgm().log(f" -> out_shape = {list(out_shape)}")
		lgm().log(f" -> out_chans = {out_chans}")
		lgm().log(f" -> embed_chans = {embed_chans}")
		lgm().log(f" -> num_layers = {num_layers}")
		lgm().log(f" -> encoder_layers = {encoder_layers}")
		lgm().log(f" -> use_mlp = {use_mlp}")
		lgm().log(f" -> mlp_ratio = {mlp_ratio}")
		lgm().log(f" -> drop_rate = {drop_rate}")
		lgm().log(f" -> drop_path_rate = {drop_path_rate}")
		lgm().log(f" -> normalization_layer = {normalization_layer}")
		lgm().log(f" -> hard_thresholding_fraction = {hard_thresholding_fraction}")
		lgm().log(f" -> big_skip = {big_skip}")
		lgm().log(f" -> factorization = {factorization}")
		lgm().log(f" -> separable = {separable}")
		lgm().log(f" -> rank = {rank}")
		lgm().log(f" -> pos_embed = {pos_embed}")

		self.spectral_transform = spectral_transform
		self.operator_type = operator_type
		self.in_shape = in_shape
		self.out_shape = out_shape
		self.grid = grid
		self.in_chans = in_chans
		self.out_chans = out_chans
		self.embed_chans = embed_chans
		self.num_layers = num_layers
		self.hard_thresholding_fraction = hard_thresholding_fraction
		self.normalization_layer = normalization_layer
		self.use_mlp = use_mlp
		self.encoder_layers = encoder_layers
		self.big_skip = big_skip
		self.factorization = factorization
		self.separable = separable,
		self.rank = rank
		self.task_type: TaskType = TaskType(cfg().task.task_type)

		# activation function
		if activation_function == "relu":
			self.activation_function = nn.ReLU
		elif activation_function == "gelu":
			self.activation_function = nn.GELU
		# for debugging purposes
		elif activation_function == "identity":
			self.activation_function = nn.Identity
		else:
			raise ValueError(f"Unknown activation function {activation_function}")

		self.pos_drop = nn.Dropout(p=drop_rate) if drop_rate > 0. else nn.Identity()
		dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.num_layers)]

		# pick norm layer
		if self.normalization_layer == "layer_norm":
			norm_layer0 = partial( nn.LayerNorm, normalized_shape=self.out_shape, eps=1e-6 )
			norm_layer1 = partial( nn.LayerNorm, normalized_shape=self.in_shape,  eps=1e-6 )
		elif self.normalization_layer == "instance_norm":
			norm_layer0 = partial(nn.InstanceNorm2d, num_features=self.embed_chans, eps=1e-6, affine=True, track_running_stats=False)
			norm_layer1 = partial(nn.InstanceNorm2d, num_features=self.embed_chans, eps=1e-6, affine=True, track_running_stats=False)
		elif self.normalization_layer == "none":
			norm_layer0 = nn.Identity
			norm_layer1 = norm_layer0
		else:
			raise NotImplementedError(f"Error, normalization {self.normalization_layer} not implemented.")

		if pos_embed == "latlon" or pos_embed == True:
			self.pos_embed = nn.Parameter(torch.zeros(1, self.embed_chans, self.in_shape[0], self.in_shape[1]))
			nn.init.constant_(self.pos_embed, 0.0)
		elif pos_embed == "lat":
			self.pos_embed = nn.Parameter(torch.zeros(1, self.embed_chans, self.in_shape[0], 1))
			nn.init.constant_(self.pos_embed, 0.0)
		elif pos_embed == "const":
			self.pos_embed = nn.Parameter(torch.zeros(1, self.embed_chans, 1, 1))
			nn.init.constant_(self.pos_embed, 0.0)
		else:
			self.pos_embed = None

		# # encoder
		# encoder_hidden_dim = int(self.embed_chans * mlp_ratio)
		# encoder = MLP(in_features = self.in_chans,
		#               out_features = self.embed_chans,
		#               hidden_features = encoder_hidden_dim,
		#               act_layer = self.activation_function,
		#               drop_rate = drop_rate,
		#               checkpointing = False)
		# self.encoder = encoder

		# construct an encoder with num_encoder_layers
		num_encoder_layers = cfg().model.encoder_layers
		encoder_hidden_dim = int(self.embed_chans * mlp_ratio)
		current_dim = self.in_chans
		encoder_layers = []

		lgm().log(f" -> num_encoder_layers = {num_encoder_layers}")
		lgm().log(f" -> encoder_hidden_dim = {encoder_hidden_dim}")
		lgm().log(f" -> current_dim = {current_dim}")

		for l in range(num_encoder_layers - 1):
			fc = nn.Conv2d(current_dim, encoder_hidden_dim, 1, bias=True)
			# initialize the weights correctly
			scale = math.sqrt(2. / current_dim)
			nn.init.normal_(fc.weight, mean=0., std=scale)
			nn.init.constant_(fc.bias, 0.0)
			encoder_layers.append(fc)
			encoder_layers.append(self.activation_function())
			current_dim = encoder_hidden_dim
		self.efc = nn.Conv2d(current_dim, self.embed_chans, 1, bias=True)
		scale = math.sqrt(1. / current_dim)
		nn.init.normal_(self.efc.weight, mean=0., std=scale)
		nn.init.constant_(self.efc.bias, 0.0)
		encoder_layers.append(self.efc)
		self.encoder = nn.Sequential(*encoder_layers)

		up_modes_lat = int( self.in_shape[0] )
		up_modes_lon = int( self.in_shape[1] // 2 + 1 )
	#	up_modes_lat = up_modes_lon = min(up_modes_lat, up_modes_lon)

		self.trans_first =  RealSHT(        *self.in_shape,     grid=self.grid).float()
		self.itrans_last =  InverseRealSHT( *self.out_shape, up_modes_lat, up_modes_lon, grid=self.grid).float()
		self.trans =        RealSHT(        *self.in_shape,  grid="legendre-gauss").float()
		self.itrans =       InverseRealSHT( *self.in_shape,  lmax=self.in_shape[0], grid="legendre-gauss").float()

		self.blocks = nn.ModuleList([])
		for i in range(self.num_layers):
			first_layer = i == 0
			last_layer = i == self.num_layers - 1

			forward_transform = self.trans_first if first_layer else self.trans
			inverse_transform = self.itrans_last if last_layer else self.itrans

			inner_skip = cfg().model.get( 'inner_skip', "none" )
			outer_skip = cfg().model.get( 'outer_skip', "identity" )

			if first_layer:
				norm_layer = norm_layer1
			elif last_layer:
				norm_layer = norm_layer0
			else:
				norm_layer = norm_layer1

			block = SphericalFourierNeuralOperatorBlock(
				forward_transform,
				inverse_transform,
				self.embed_chans,
				self.embed_chans,
				operator_type=self.operator_type,
				mlp_ratio=mlp_ratio,
				drop_rate=drop_rate,
				drop_path=dpr[i],
				act_layer=self.activation_function,
				norm_layer=norm_layer,
				inner_skip=inner_skip,
				outer_skip=outer_skip,
				use_mlp=use_mlp,
				factorization=self.factorization,
				separable=self.separable,
				rank=self.rank)

			self.blocks.append(block)

		# # decoder
		# decoder_hidden_dim = int(self.embed_chans * mlp_ratio)
		# self.decoder = MLP(in_features = self.embed_chans + self.big_skip*self.in_chans,
		#                    out_features = self.out_chans,
		#                    hidden_features = decoder_hidden_dim,
		#                    act_layer = self.activation_function,
		#                    drop_rate = drop_rate,
		#                    checkpointing = False)

		# construct an decoder with num_decoder_layers
		num_decoder_layers = 1
		decoder_hidden_dim = int(self.embed_chans * mlp_ratio)
		current_dim = self.embed_chans + self.big_skip * self.in_chans
		decoder_layers = []
		for l in range(num_decoder_layers - 1):
			fc = nn.Conv2d(current_dim, decoder_hidden_dim, 1, bias=True)
			# initialize the weights correctly
			scale = math.sqrt(2. / current_dim)
			nn.init.normal_(fc.weight, mean=0., std=scale)
			if fc.bias is not None:
				nn.init.constant_(fc.bias, 0.0)
			decoder_layers.append(fc)
			decoder_layers.append(self.activation_function())
			current_dim = decoder_hidden_dim
		self.dfc = nn.Conv2d(current_dim, self.out_chans, 1, bias=False)
		scale = math.sqrt(1. / current_dim)
		nn.init.normal_(self.dfc.weight, mean=0., std=scale)
		if self.dfc.bias is not None:
			nn.init.constant_(self.dfc.bias, 0.0)
		decoder_layers.append(self.dfc)
		self.decoder = nn.Sequential(*decoder_layers)

	@torch.jit.ignore
	def no_weight_decay(self):
		return {"pos_embed", "cls_token"}

	def forward_features(self, x):
		x = self.pos_drop(x)

		for blk in self.blocks:
			x = blk(x)

		return x

	def forward(self, x):
		lgm().log(f"Forward: x{tuple(x.shape)}, %N={pctnant(x)}")
		residual = x
		x = self.encoder(x)
		lgm().log( f"Embed: {tuple(residual.shape)} -> {tuple(x.shape)}, W{tuple(self.efc.weight.shape)}, pos_embed{tuple(self.pos_embed.shape)}")

		if self.pos_embed is not None:
			lgm().log(f"Pos Embed: pos_embed{tuple(self.pos_embed.shape)} + x{tuple(x.shape)},")
			x = x + self.pos_embed

		x = self.forward_features(x)
		lgm().log( f"FF: x.shape={tuple(x.shape)}, %N={pctnant(x)}")

		if self.big_skip:
			x = torch.cat((x, residual), dim=1)

		residual = x
		x = self.decoder(x)
		lgm().log(f"Decode: {tuple(residual.shape)} -> {tuple(x.shape)}, W{tuple(self.dfc.weight.shape)}")

		return x


