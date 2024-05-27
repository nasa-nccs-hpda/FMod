import torch, math
import xarray, traceback, random
from datetime import datetime
from torch import Tensor
from typing import Any, Dict, List, Tuple, Union, Sequence
from fmod.base.util.config import cdelta, cfg, cval, get_data_coords
from fmod.base.util.grid import GridOps
from fmod.base.util.array import array2tensor
import torch_harmonics as harmonics
from fmod.data.batch import BatchDataset
from fmod.model.sres.manager import SRModels
from fmod.base.util.logging import lgm, exception_handled
from fmod.base.util.ops import pctnan, pctnant
from fmod.controller.checkpoints import CheckpointManager
from enum import Enum
import numpy as np, xarray as xa
from fmod.controller.stats import l2loss
import torch.nn as nn
import time

Tensors = Sequence[Tensor]
TensorOrTensors = Union[Tensor, Tensors]

class LearningContext(Enum):
	Training = 'train'
	Validation = 'val'

def smean( data: xarray.DataArray, dims: List[str] = None ) -> str:
	means: np.ndarray = data.mean(dim=dims).values
	return str( [ f"{mv:.2f}" for mv in means ] )

def sstd( data: xarray.DataArray, dims: List[str] = None ) -> str:
	stds: np.ndarray = data.std(dim=dims).values
	return str( [ f"{mv:.2f}" for mv in stds ] )

def log_stats( name: str, data: xarray.DataArray, dims: List[str], display: bool = True):
	lgm().log(f' * {name} mean: {smean(data, dims)}', display=display)
	lgm().log(f' * {name} std:  { sstd(data, dims)}', display=display)

def mse( data: xarray.DataArray, target: xarray.DataArray, dims: List[str] ) -> xarray.DataArray:
	sdiff: xarray.DataArray = (target - data)**2
	return np.sqrt( sdiff.mean(dim=dims) )

def batch( members: List[xarray.DataArray] ) -> xarray.DataArray:
	return xarray.concat( members, dim="batch" )

def npa( tensor: Tensor ) -> np.ndarray:
	return tensor.detach().cpu().numpy()

def fmtfl( flist: List[float] ) -> str:
	svals = ','.join( [ f"{fv:.4f}" for fv in flist ] )
	return f"[{svals}]"

def shuffle( data: Tensor ) -> Tensor:
	idx = torch.randperm(data.shape[0])
	return data[idx,...]

def tas( ta: Any ) -> str:
	return list(ta) if (type(ta) is torch.Size) else ta
def ts( t: Tensor ) -> str:
	return f"{tas(t.dim())}{tas(t.shape)}"

class TileGrid(object):

	def __init__(self, context: LearningContext = LearningContext.Training):
		self.context = context
		cfg_origin = "origin" if LearningContext == LearningContext.Training else "val_origin"
		cfg_tgrid  = "tile_grid" if LearningContext == LearningContext.Training else "val_tile_grid"
		self.origin: Dict[str,int] = cfg().task.get( cfg_origin, dict(x=0,y=0) )
		self.tile_size: Dict[str,int] = cfg().task.tile_size
		self.tile_grid: Dict[str, int] = cfg().task.get( cfg_tgrid, dict(x=1,y=1) )
		self.tlocs: List[Dict[str,int]] = []
		downscale_factors: List[int] = cfg().model.downscale_factors
		self.downscale_factor = math.prod(downscale_factors)

	def get_tile_origin( self, ix: int, iy: int ) -> Dict[str, int]:
		return {d: self.origin[d] + self.cdim(ix, iy, d) * self.tile_size[d] for d in ['x', 'y']}

	def get_tile_locations(self, randomize=False) -> List[Dict[str, int]]:
		if len(self.tlocs) == 0:
			for ix in range(self.tile_grid['x']):
				for iy in range(self.tile_grid['y']):
					self.tlocs.append( self.get_tile_origin(ix,iy) )
		if randomize: random.shuffle(self.tlocs)
		return self.tlocs

	@classmethod
	def cdim(cls, ix: int, iy: int, dim: str) -> int:
		if dim == 'x': return ix
		if dim == 'y': return iy

def downscale(self, origin: Dict[str,int] ):
	return { d: v*self.downscale_factor for d,v in origin.items() }

class ModelTrainer(object):

	model_cfg = ['batch_size', 'num_workers', 'persistent_workers' ]

	def __init__(self, model_manager: SRModels ):
		super(ModelTrainer, self).__init__()
		self.input_dataset: BatchDataset = model_manager.datasets['input']
		self.target_dataset: BatchDataset = model_manager.datasets['target']
		self.device: torch.device = model_manager.device
		self.model_manager = model_manager
		self.min_loss = float('inf')
		self.eps = 1e-6
		self._sht, self._isht = None, None
		self.scheduler = None
		self.optimizer = None
		self.checkpoint_manager = None
		self.model = model_manager.get_model()
		self.checkpoint_manager: CheckpointManager = None
		self.loss_module: nn.Module = None
		self.layer_losses = []
		self.channel_idxs: torch.LongTensor = None
		self.target_variables = cfg().task.target_variables
		self.train_dates = self.input_dataset.train_dates
		self.downscale_factors = cfg().model.downscale_factors
		self.scale_factor = math.prod(self.downscale_factors)
		self.upsampler = nn.UpsamplingBilinear2d(scale_factor=self.scale_factor)
		self.conform_to_data_grid()
		self.grid_shape, self.gridops, self.lmax = self.configure_grid()
		self.current_input: torch.Tensor = None
		self.current_upsampled: torch.Tensor = None
		self.current_target: torch.Tensor = None
		self.current_product: TensorOrTensors = None

	def upsample(self, tensor: Tensor, renorm: bool = True ) -> Tensor:
		upsampled = self.upsampler(tensor)
		if renorm:
			if tensor.ndim == 3: upsampled = torch.unsqueeze(upsampled, 1)
			upsampled = nn.functional.normalize( upsampled )
		return upsampled

	def configure_grid(self):
		tar: xarray.DataArray = self.target_dataset.get_current_batch_array()
		grid_shape = tar.shape[-2:]
		gridops = GridOps(*grid_shape)
		lgm().log(f"SHAPES: target{list(tar.shape)}, (nlat, nlon)={grid_shape}")
		lmax = tar.shape[-2]
		return grid_shape, gridops, lmax

	def conform_to_data_grid(self, **kwargs):
		if cfg().task.conform_to_grid:
			data: xarray.DataArray = self.input_dataset.get_current_batch()['input']
			data_origin: Dict[str, float] = get_data_coords(data, cfg().task['origin'])
			dc = cdelta(data)
			lgm().log(f"  ** snap_origin_to_data_grid: {cfg().task['origin']} -> {data_origin}", **kwargs)
			cfg().task['origin'] = data_origin
			cfg().task['extent'] = {dim: float(cval(data, dim, -1) + dc[cfg().task.coords[dim]]) for dim in data_origin.keys()}
			print(f" *** conform_to_data_grid: origin={cfg().task['origin']} extent={cfg().task['extent']} *** ")

	@property
	def sht(self):
		if self._sht is None:
			self._sht = harmonics.RealSHT(*self.grid_shape, lmax=self.lmax, mmax=self.lmax, grid='equiangular', csphase=False)
		return self._sht

	@property
	def isht(self):
		if self._isht is None:
			self._isht = harmonics.InverseRealSHT( *self.grid_shape, lmax=self.lmax, mmax=self.lmax, grid='equiangular', csphase=False)
		return self._isht

	def tensor(self, data: xarray.DataArray) -> torch.Tensor:
		return Tensor(data.values).to(self.device)

	@property
	def loader_args(self) -> Dict[str, Any]:
		return { k: cfg().model.get(k) for k in self.model_cfg }

	def l2loss_sphere(self, prd: torch.Tensor, tar: torch.Tensor, relative=False, squared=True) -> torch.Tensor:
		loss = self.gridops.integrate_grid((prd - tar) ** 2, dimensionless=True).sum(dim=-1)
		if relative:
			loss = loss / self.gridops.integrate_grid(tar ** 2, dimensionless=True).sum(dim=-1)

		if not squared:
			loss = torch.sqrt(loss)
		loss = loss.mean()

		return loss

	def spectral_l2loss_sphere(self, prd: torch.Tensor, tar: torch.Tensor, relative=False, squared=True) -> torch.Tensor:
		# compute coefficients
		coeffs = torch.view_as_real(self.sht(prd - tar))
		coeffs = coeffs[..., 0] ** 2 + coeffs[..., 1] ** 2
		norm2 = coeffs[..., :, 0] + 2 * torch.sum(coeffs[..., :, 1:], dim=-1)
		loss = torch.sum(norm2, dim=(-1, -2))

		if relative:
			tar_coeffs = torch.view_as_real(self.sht(tar))
			tar_coeffs = tar_coeffs[..., 0] ** 2 + tar_coeffs[..., 1] ** 2
			tar_norm2 = tar_coeffs[..., :, 0] + 2 * torch.sum(tar_coeffs[..., :, 1:], dim=-1)
			tar_norm2 = torch.sum(tar_norm2, dim=(-1, -2))
			loss = loss / tar_norm2

		if not squared:
			loss = torch.sqrt(loss)
		loss = loss.mean()
		return loss

	def charbonnier(self, prd: torch.Tensor, tar: torch.Tensor) -> torch.Tensor:
		error = torch.sqrt( ((prd - tar) ** 2) + self.eps )
		return torch.mean(error)

	def single_product_loss(self, prd: torch.Tensor, tar: torch.Tensor) -> torch.Tensor:
		#	print( f" ----->> single_product_loss: prd{prd.shape} -- tar{tar.shape}")
		if cfg().model.loss_fn == 'l2':
			loss = l2loss(prd, tar)
		elif cfg().model.loss_fn == 'l2s':
			loss = self.l2loss_sphere(prd, tar)
		elif cfg().model.loss_fn == "spectral-l2s":
			loss = self.spectral_l2loss_sphere(prd, tar)
		elif cfg().model.loss_fn == "charbonnier":
			loss = self.charbonnier(prd, tar)
		else:
			raise Exception("Unknown single-product loss function {}".format(cfg().model.loss_fn))
		return loss

	def get_multiscale_targets(self, hr_targ: Tensor) -> List[Tensor]:
		targets: List[Tensor] = [hr_targ]
		for usf in self.downscale_factors[:-1]:
			targets.append( torch.nn.functional.interpolate(targets[-1], scale_factor=1.0/usf, mode='bilinear') )
		targets.reverse()
		return targets

	def loss(self, products: TensorOrTensors, target: Tensor ) -> torch.Tensor:
		loss, ptype, self.layer_losses = None, type(products), []
		if ptype == torch.Tensor:
			loss = self.single_product_loss( products, target)
		elif not cfg().model.get('multiscale_loss',False):
			loss = self.single_product_loss(products[-1], target)
		else:
			targets: List[Tensor] = self.get_multiscale_targets(target)
			#	print(f"  Output Shapes: { ','.join([str(list(out.shape)) for out in products]) }")
			#	print(f"  Target Shapes: { ','.join([str(list(tar.shape)) for tar in targets]) }")
			for iL, (layer_output, layer_target) in enumerate( zip(products,targets)):
				layer_loss = self.single_product_loss(layer_output, layer_target)
				#		print( f"Layer-{iL}: Output{list(layer_output.shape)}, Target{list(layer_target.shape)}, loss={layer_loss.item():.5f}")
				loss = layer_loss if (loss is None) else (loss + layer_loss)
				self.layer_losses.append( layer_loss.item() )
		#	print( f" --------- Layer losses: {layer_losses} --------- ")
		return loss

	def get_batch(self, origin: Dict[str,int], batch_date: datetime, as_tensor: bool = True, shuffle: bool = True ) -> Dict[str,Union[torch.Tensor,xarray.DataArray]]:
		input_batch: Dict[str, xarray.DataArray]  = self.input_dataset.get_batch(origin,batch_date)
		target_batch: Dict[str, xarray.DataArray] = self.target_dataset.get_batch(origin,batch_date)
		binput: xarray.DataArray = input_batch['input']
		btarget: xarray.DataArray = target_batch['target']
		if shuffle:
			batch_perm: Tensor = torch.randperm(binput.shape[0])
			binput: xarray.DataArray = binput[ batch_perm, ... ]
			btarget: xarray.DataArray = btarget[ batch_perm, ... ]
		lgm().log(f" *** input{binput.dims}{binput.shape}, pct-nan= {pctnan(binput.values)}")
		lgm().log(f" *** target{btarget.dims}{btarget.shape}, pct-nan= {pctnan(btarget.values)}")
		if as_tensor:  return dict( input=array2tensor(binput), target=array2tensor(btarget) )
		else:          return dict( input=binput,               target=btarget )

	def get_srbatch(self, origin: Dict[str,int], batch_date: datetime, as_tensor: bool = True, shuffle: bool = True ) -> Dict[str,Union[torch.Tensor,xarray.DataArray]]:
		binput:  xarray.DataArray  = self.input_dataset.get_batch_array(origin,batch_date)
		btarget:  xarray.DataArray = self.target_dataset.get_batch_array(origin,batch_date)
		if shuffle:
			batch_perm: Tensor = torch.randperm(binput.shape[0])
			binput: xarray.DataArray = binput[ batch_perm, ... ]
			btarget: xarray.DataArray = btarget[ batch_perm, ... ]
		lgm().log(f" *** input{binput.dims}{binput.shape}, mean={binput.mean():.3f}, std={binput.std():.3f}")
		lgm().log(f" *** target{btarget.dims}{btarget.shape}, mean={btarget.mean():.3f}, std={btarget.std():.3f}")
		if as_tensor:  return dict( input=array2tensor(binput), target=array2tensor(btarget) )
		else:          return dict( input=binput,               target=btarget )

	def get_current_input(self, targets_only: bool = True ) -> np.ndarray:
		if self.current_input is not None:
			curr_input: Tensor = self.get_target_channels(self.current_input) if targets_only else self.current_input
			return npa( curr_input )

	def get_current_upsampled(self, targets_only: bool = True) -> np.ndarray:
		if self.current_upsampled is not None:
			curr_upsampled: Tensor = self.get_target_channels(self.current_upsampled) if targets_only else self.current_upsampled
			return npa( curr_upsampled )

	def get_current_target(self) -> np.ndarray:
		return None if (self.current_target is None) else npa( self.current_target )

	def get_current_product(self) -> np.ndarray:
		return None if (self.current_product is None) else npa( self.current_product )

	@exception_handled
	def train(self, **kwargs ):
		seed = kwargs.get('seed',333)
		load_state = kwargs.get( 'load_state', '' )
		save_state = kwargs.get('save_state', True)

		torch.manual_seed(seed)
		torch.cuda.manual_seed(seed)
		self.scheduler = kwargs.get( 'scheduler', None )
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg().task.lr, weight_decay=cfg().task.get('weight_decay',0.0))
		self.checkpoint_manager = CheckpointManager(self.model,self.optimizer)
		epoch0, nepochs, batch_iter = 0, cfg().task.nepochs, cfg().task.batch_iter
		loss: torch.Tensor = torch.tensor(float('inf'), device=self.device)
		train_start = time.time()
		if load_state:
			train_state = self.checkpoint_manager.load_checkpoint(load_state)
			epoch0 = train_state.get('epoch',0)
			nepochs += epoch0
		else:
			print( " *** No checkpoint loaded: training from scratch *** ")

		for epoch in range(epoch0,nepochs):
			print(f'Epoch {epoch + 1}/{nepochs}: ')
			epoch_start = time.time()
			self.optimizer.zero_grad(set_to_none=True)
			lgm().log(f"  ----------- Epoch {epoch + 1}/{nepochs}   ----------- ", display=True )

			self.model.train()
			batch_dates: List[datetime] = self.input_dataset.get_batch_dates()
			tile_locs: List[Dict[str,int]] =  TileGrid( LearningContext.Training ).get_tile_locations()
			for batch_date in batch_dates:
				for tile_loc in tile_locs:
					try:
						train_data: Dict[str,Tensor] = self.get_srbatch(tile_loc,batch_date)
						inp: Tensor = train_data['input']
						target: Tensor   = train_data['target']
						for biter in range(batch_iter):
							prd, targ = self.apply_network( inp, target )

							lgm().log( f"apply_network: inp{ts(inp)} target{ts(target)} prd{ts(prd)} targ{ts(targ)}")
							loss = self.loss( prd, targ )
							lgm().log(f" ** Loss({batch_date}:{biter}:[{tile_loc['y']:3d},{tile_loc['x']:3d}] {list(prd.shape)}->{list(targ.shape)}:  {loss.item():.5f}  {fmtfl(self.layer_losses)}", display=True, end="" )
							self.current_input = inp
							self.current_upsampled = self.upsample(inp)
							self.current_target = targ
							self.current_product = prd
							self.optimizer.zero_grad(set_to_none=True)
							loss.backward()
							self.optimizer.step()

						if save_state:
							self.checkpoint_manager.save_checkpoint( epoch, loss.item() )
					except Exception as e:
						print( f"\n !!!!! Error processing tile_loc={tile_loc}, batch_date={batch_date} !!!!! {e}")
						print( traceback.format_exc() )

			if self.scheduler is not None:
				self.scheduler.step()

			epoch_time = time.time() - epoch_start
			lgm().log(f'Epoch {epoch}, time: {epoch_time:.1f}, loss: {loss.item():.5f} {fmtfl(self.layer_losses)}', display=True)

		train_time = time.time() - train_start

		print(f'--------------------------------------------------------------------------------')
		ntotal_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
		print(f' -------> Training model with {ntotal_params} took {train_time/60:.2f} min.')

		return loss.item()

	def apply_network( self, input_data: Tensor, target_data: Tensor = None ) -> Tuple[TensorOrTensors,Tensor]:
		net_input: Tensor  = input_data
		target: Tensor = target_data
		product: TensorOrTensors = self.model( net_input )
		if type(product) == torch.Tensor:
			result  =  self.get_target_channels(product)
		#		print( f"get_train_target, input shape={input_data.shape}, product shape={product.shape}, output shape={result.shape}, channel_idxs={channel_idxs}")
		else:
			result = [ self.get_target_channels(prod) for prod in product ]
		#		print(f"get_train_target, input shape={input_data.shape}, product shape={product[0].shape}, output shape={result[0].shape}, channel_idxs={channel_idxs}")
		net_target = self.get_target_channels( target )
		return result, net_target

	def get_target_channels(self, batch_data: Tensor) -> Tensor:
		if self.channel_idxs is None:
			cidxs: List[int] = self.input_dataset.get_channel_idxs(self.target_variables)
			self.channel_idxs = torch.LongTensor( cidxs ).to( self.device )
		channels = [ torch.select(batch_data, 1, cidx ) for cidx in self.channel_idxs ]
		return channels[0] if (len(channels) == 1) else torch.cat( channels, dim = 1 )

	def forecast(self, **kwargs ) -> Tuple[ List[np.ndarray], List[np.ndarray], List[np.ndarray] ]:
		seed = kwargs.get('seed',0)
		max_step = kwargs.get('max_step',5)
		torch.manual_seed(seed)
		torch.cuda.manual_seed(seed)
		inputs, predictions, targets = [], [], []
		with torch.inference_mode():
			for istep, batch_data in enumerate(self.data_iter):
				inp: torch.Tensor = array2tensor(batch_data['input'])
				tar: torch.Tensor = array2tensor(batch_data['target'])
				if istep == max_step: break
				out: Tensor = self.model(inp)
				lgm().log(f' * STEP {istep}, in: [{list(inp.shape)}, {pctnant(inp)}], out: [{list(out.shape)}, {pctnant(out)}]')
				predictions.append( npa(out) )
				targets.append( npa(tar) )
				inputs.append( npa(inp) )
		lgm().log(f' * INFERENCE complete, #predictions={len(predictions)}, target: {targets[0].shape}', display=True )
		for input1, prediction, target in zip(inputs,predictions,targets):
			lgm().log(f' ---> *** input: {input1.shape}, pctnan={pctnan(input1)} *** prediction: {prediction.shape}, pctnan={pctnan(prediction)} *** target: {target.shape}, pctnan={pctnan(target)}')

		return inputs, targets, predictions

	# def apply1(self, origin: Dict[str,int], batch_date: datetime, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	# 	seed = kwargs.get('seed',0)
	# 	torch.manual_seed(seed)
	# 	torch.cuda.manual_seed(seed)
	# 	with torch.inference_mode():
	# 		input_batch = self.input_dataset.get_batch( origin, batch_date)
	# 		target_batch = self.target_dataset.get_batch( origin, batch_date)
	# 		inp: torch.Tensor = array2tensor( input_batch['input'] )
	# 		tar: torch.Tensor = array2tensor( target_batch['target'] )
	# 		out: TensorOrTensors = self.apply_network(inp)
	# 		product: torch.Tensor = out if type(out) is torch.Tensor else out[-1]
	# 		lgm().log(f' * in: {list(inp.shape)}, target: {list(tar.shape)}, out: {list(product.shape)}', display=True)
	# 		return npa(inp), npa(tar), npa(product)

	def apply(self, tile_loc: Dict[str, int], batch_date: datetime, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
		seed = kwargs.get('seed', 0)
		torch.manual_seed(seed)
		torch.cuda.manual_seed(seed)
		with torch.inference_mode():
			train_data: Dict[str, Tensor] = self.get_srbatch(tile_loc, batch_date)
			inp: Tensor = train_data['input'].squeeze()
			target: Tensor = train_data['target'].squeeze()
			prd,targ = self.apply_network(inp,target)
			loss: torch.Tensor = self.loss(prd, targ)
			filtered_input: np.ndarray = self.model_manager.filter_targets(npa(inp))
			return filtered_input,  npa(targ),  npa(prd), loss.item()
