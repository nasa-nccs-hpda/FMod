import torch, math
import xarray, traceback, random
from datetime import datetime
from torch import Tensor
from typing import Any, Dict, List, Tuple, Union, Sequence, Optional
from fmod.base.io.loader import ncFormat, TSet
from fmod.base.source.loader import srRes
from fmod.base.io.loader import TSet
from fmod.base.util.config import cdelta, cfg, cval, get_data_coords, dateindex
#from fmod.base.util.grid import GridOps
from fmod.base.util.array import array2tensor
from fmod.model.sres.mscnn.network import Upsampler
from fmod.data.batch import BatchDataset, TileGrid
from fmod.model.sres.manager import SRModels, ResultsAccumulator
from fmod.base.util.logging import lgm
from fmod.base.util.ops import pctnan, pctnant
from fmod.controller.checkpoints import CheckpointManager
import numpy as np, xarray as xa
from fmod.controller.stats import l2loss
import torch.nn as nn
import time, csv

Tensors = Sequence[Tensor]
TensorOrTensors = Union[Tensor, Tensors]
MLTensors = Dict[ TSet, torch.Tensor]

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

def npa( ts: TensorOrTensors ) -> np.ndarray:
	t = ts if type(ts) == Tensor else ts[-1]
	return t.detach().cpu().numpy()

def fmtfl( flist: List[float] ) -> str:
	svals = ','.join( [ f"{fv:.4f}" for fv in flist ] )
	return f"[{svals}]"

def shuffle( data: Tensor ) -> Tensor:
	idx = torch.randperm(data.shape[0])
	return data[idx,...]

def tas( ta: Any ) -> str:
	return list(ta) if (type(ta) is torch.Size) else ta

def ts( ts: TensorOrTensors ) -> str:
	if type(ts) == torch.Tensor: return tas(ts.shape)
	else:                        return str( [ tas(t.shape) for t in ts ] )

def unsqueeze( tensor: Tensor ) -> Tensor:
	if tensor.ndim == 2:
		tensor = torch.unsqueeze(tensor, 0)
		tensor = torch.unsqueeze(tensor, 0)
	elif tensor.ndim == 3:
		tensor = torch.unsqueeze(tensor, 1)
	return tensor

def normalize( tensor: Tensor ) -> Tensor:
	tensor = unsqueeze( tensor )
	tensor = tensor - tensor.mean(dim=[2,3], keepdim=True)
	return tensor / tensor.std(dim=[2,3], keepdim=True)

def downscale(self, origin: Dict[str,int] ):
	return { d: v*self.downscale_factor for d,v in origin.items() }

class ModelTrainer(object):

	model_cfg = ['batch_size', 'num_workers', 'persistent_workers' ]

	def __init__(self, model_manager: SRModels, results_accumulator: ResultsAccumulator = None ):
		super(ModelTrainer, self).__init__()
		self.device: torch.device = model_manager.device
		self.model_manager = model_manager
		self.results_accum = results_accumulator
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
		self.train_dates: List[datetime] = self.input_dataset(TSet.Train).train_dates
		self.downscale_factors = cfg().model.downscale_factors
		self.scale_factor = math.prod(self.downscale_factors)
		self.upsampler: Upsampler = Upsampler( downscale_factors=cfg().model.downscale_factors, mode=cfg().model.ups_mode )
		self.conform_to_data_grid()
	#	self.grid_shape, self.gridops, self.lmax = self.configure_grid()
		self.input: MLTensors = {}
		self.target: MLTensors = {}
		self.product: MLTensors = {}
		self.interp: MLTensors = {}
		self.current_losses: Dict[str,float] = {}
		self.time_index: int = -1
		self.tile_index: Optional[Tuple[int,int]] = None
		self.best_loss: Dict[TSet,float] = { tset: float('inf') for tset in TSet }

	@property
	def model_name(self):
		return self.model_manager.model_name

	def input_dataset(self, tset: TSet)-> BatchDataset:
		return self.model_manager.get_dataset( srRes.Low, tset )

	def target_dataset(self, tset: TSet)-> BatchDataset:
		return self.model_manager.get_dataset(srRes.High, tset )

	def get_sample_input(self, tset: TSet, targets_only: bool = True) -> xa.DataArray:
		return self.model_manager.get_sample_input( tset, targets_only )

	def get_sample_target(self, tset: TSet) -> xa.DataArray:
		return self.model_manager.get_sample_target(tset)

	def upsample(self, tensor: Tensor ) -> Tensor:
		upsampled = self.upsampler( unsqueeze( tensor ) )
		return upsampled

	# def configure_grid(self, tset: TSet ):
	# 	tar: xarray.DataArray = self.target_dataset(tset).get_current_batch_array()
	# 	grid_shape = tar.shape[-2:]
	# 	gridops = GridOps(*grid_shape,self.device)
	# 	lgm().log(f"SHAPES: target{list(tar.shape)}, (nlat, nlon)={grid_shape}")
	# 	lmax = tar.shape[-2]
	# 	return grid_shape, gridops, lmax

	def conform_to_data_grid(self, **kwargs):
		if cfg().task.conform_to_grid:
			data: xarray.DataArray = self.input_dataset(TSet.Train).get_current_batch_array()
			data_origin: Dict[str, float] = get_data_coords(data, cfg().task['origin'])
			dc = cdelta(data)
			lgm().log(f"  ** snap_origin_to_data_grid: {cfg().task['origin']} -> {data_origin}", **kwargs)
			cfg().task['origin'] = data_origin
			cfg().task['extent'] = {dim: float(cval(data, dim, -1) + dc[cfg().task.coords[dim]]) for dim in data_origin.keys()}
			print(f" *** conform_to_data_grid: origin={cfg().task['origin']} extent={cfg().task['extent']} *** ")

	# @property
	# def sht(self):
	# 	if self._sht is None:
	# 		self._sht = harmonics.RealSHT(*self.grid_shape, lmax=self.lmax, mmax=self.lmax, grid='equiangular', csphase=False)
	# 	return self._sht
	#
	# @property
	# def isht(self):
	# 	if self._isht is None:
	# 		self._isht = harmonics.InverseRealSHT( *self.grid_shape, lmax=self.lmax, mmax=self.lmax, grid='equiangular', csphase=False)
	# 	return self._isht

	def tensor(self, data: xarray.DataArray) -> torch.Tensor:
		return Tensor(data.values).to(self.device)

	@property
	def loader_args(self) -> Dict[str, Any]:
		return { k: cfg().model.get(k) for k in self.model_cfg }

	# def l2loss_sphere(self, prd: torch.Tensor, tar: torch.Tensor, relative=False, squared=True) -> torch.Tensor:
	# 	loss = self.gridops.integrate_grid((prd - tar) ** 2, dimensionless=True).sum(dim=-1)
	# 	if relative:
	# 		loss = loss / self.gridops.integrate_grid(tar ** 2, dimensionless=True).sum(dim=-1)
	#
	# 	if not squared:
	# 		loss = torch.sqrt(loss)
	# 	loss = loss.mean()
	#
	# 	return loss

	# def spectral_l2loss_sphere(self, prd: torch.Tensor, tar: torch.Tensor, relative=False, squared=True) -> torch.Tensor:
	# 	# compute coefficients
	# 	coeffs = torch.view_as_real(self.sht(prd - tar))
	# 	coeffs = coeffs[..., 0] ** 2 + coeffs[..., 1] ** 2
	# 	norm2 = coeffs[..., :, 0] + 2 * torch.sum(coeffs[..., :, 1:], dim=-1)
	# 	loss = torch.sum(norm2, dim=(-1, -2))
	#
	# 	if relative:
	# 		tar_coeffs = torch.view_as_real(self.sht(tar))
	# 		tar_coeffs = tar_coeffs[..., 0] ** 2 + tar_coeffs[..., 1] ** 2
	# 		tar_norm2 = tar_coeffs[..., :, 0] + 2 * torch.sum(tar_coeffs[..., :, 1:], dim=-1)
	# 		tar_norm2 = torch.sum(tar_norm2, dim=(-1, -2))
	# 		loss = loss / tar_norm2
	#
	# 	if not squared:
	# 		loss = torch.sqrt(loss)
	# 	loss = loss.mean()
	# 	return loss

	def charbonnier(self, prd: torch.Tensor, tar: torch.Tensor) -> torch.Tensor:
		error = torch.sqrt( ((prd - tar) ** 2) + self.eps )
		return torch.mean(error)

	def single_product_loss(self, prd: torch.Tensor, tar: torch.Tensor) -> torch.Tensor:
		#	print( f" ----->> single_product_loss: prd{prd.shape} -- tar{tar.shape}")
		if cfg().model.loss_fn == 'l2':
			loss = l2loss(prd, tar)
		# elif cfg().model.loss_fn == 'l2s':
		# 	loss = self.l2loss_sphere(prd, tar)
		# elif cfg().model.loss_fn == "spectral-l2s":
		# 	loss = self.spectral_l2loss_sphere(prd, tar)
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

	def loss(self, products: TensorOrTensors, target: Tensor ) -> Tuple[torch.Tensor,torch.Tensor]:
		sloss, mloss, ptype, self.layer_losses = None, None, type(products), []
		if ptype == torch.Tensor:
			sloss = self.single_product_loss( products, target)
			mloss = sloss
		else:
			sloss = self.single_product_loss(products[-1], target)
			targets: List[Tensor] = self.get_multiscale_targets(target)
			for iL, (layer_output, layer_target) in enumerate( zip(products,targets)):
				layer_loss = self.single_product_loss(layer_output, layer_target)
				#		print( f"Layer-{iL}: Output{list(layer_output.shape)}, Target{list(layer_target.shape)}, loss={layer_loss.item():.5f}")
				mloss = layer_loss if (mloss is None) else (mloss + layer_loss)
				self.layer_losses.append( layer_loss.item() )
		return sloss, mloss

	def get_srbatch(self, origin: Dict[str,int], tset: TSet, start_coord: Union[datetime,int], **kwargs  ) -> Dict[str,Union[torch.Tensor,xarray.DataArray]]:
		as_tensor: bool = kwargs.pop('as_tensor',True)
		shuffle: bool = kwargs.pop('shuffle',False)
		binput:  xarray.DataArray  = self.input_dataset(tset).get_batch_array(origin,start_coord,**kwargs)
		btarget:  xarray.DataArray = self.target_dataset(tset).get_batch_array(origin,start_coord,**kwargs)
		if shuffle:
			batch_perm: Tensor = torch.randperm(binput.shape[0])
			binput: xarray.DataArray = binput[ batch_perm, ... ]
			btarget: xarray.DataArray = btarget[ batch_perm, ... ]
		lgm().log(f" *** input{binput.dims}{binput.shape}, mean={binput.mean():.3f}, std={binput.std():.3f}, range=({btarget.values.min():.3f},{btarget.values.max():.3f})")
		lgm().log(f" *** target{btarget.dims}{btarget.shape}, mean={btarget.mean():.3f}, std={btarget.std():.3f}, range=({btarget.values.min():.3f},{btarget.values.max():.3f})")
		if as_tensor:  return dict( input=array2tensor(binput), target=array2tensor(btarget) )
		else:          return dict( input=binput,               target=btarget )

	def get_ml_input(self, tset: TSet, targets_only: bool = False) -> np.ndarray:
		if tset not in self.input: self.evaluate(tset)
		ml_input: Tensor = self.get_target_channels(self.input[tset.value]) if targets_only else self.input[tset.value]
		return npa( ml_input ).astype(np.float32)

	def get_ml_upsampled(self, tset: TSet) -> np.ndarray:
		inp: np.ndarray = self.get_ml_input(tset)
		ups: Tensor = self.upsample( torch.from_numpy( inp ) )
		return ups.numpy()

	def get_ml_target(self, tset: TSet) -> np.ndarray:
		if tset.value not in self.target: self.evaluate(tset)
		return npa( self.target[tset.value] )

	def get_ml_product(self, tset: TSet) -> np.ndarray:
		if tset.value not in self.product: self.evaluate(tset)
		return npa(self.product[tset.value])

	def train(self, **kwargs ) -> Dict[str,float]:
		if cfg().task['nepochs'] == 0: return {}
		refresh_state = kwargs.get( 'cppath', False )
		seed = kwargs.get( 'seed', 4456 )

		torch.manual_seed(seed)
		torch.cuda.manual_seed(seed)
		self.scheduler = kwargs.get( 'scheduler', None )
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg().task.lr, weight_decay=cfg().task.get('weight_decay',0.0))
		self.checkpoint_manager = CheckpointManager(self.model,self.optimizer)
		epoch0, epoch_loss, nepochs, batch_iter, loss_history, eval_losses, tset = 0, 0.0, cfg().task.nepochs, cfg().task.batch_iter, [], {}, TSet.Train
		train_start = time.time()
		if refresh_state:
			self.checkpoint_manager.clear_checkpoint(TSet.Train)
			print(" *** No checkpoint loaded: training from scratch *** ")
		else:
			train_state = self.checkpoint_manager.load_checkpoint(tset)
			epoch0 = train_state.get('epoch',0)
			nepochs += epoch0

		self.record_eval(epoch0)
		for epoch in range(epoch0+1,nepochs+1):
			epoch_start = time.time()
			self.optimizer.zero_grad(set_to_none=True)
			self.model.train()
			start_coords: List[Union[datetime,int]] = self.input_dataset(TSet.Train).get_batch_start_coords()
			tile_locs: Dict[ Tuple[int,int], Dict[str,int] ] =  TileGrid( TSet.Train).get_tile_locations()
			lgm().log(f"  ----------- Epoch {epoch}/{nepochs}, nbatches={len(start_coords)}   ----------- ", display=True )

			batch_losses = []
			for batch_index, start_coord in enumerate(start_coords):
				try:
					losses = torch.tensor( 0.0, device=self.device, dtype=torch.float32 )
					inp, prd, targ = None, None, None
					for tIdx, tile_loc in tile_locs.items():
						train_data: Dict[str,Tensor] = self.get_srbatch(tile_loc,TSet.Train,start_coord)
						inp = train_data['input']
						target: Tensor   = train_data['target']
						for biter in range(batch_iter):
							prd, targ = self.apply_network( inp, target )
							[sloss,mloss] = self.loss( prd, targ )
							losses += sloss
							lgm().log(f"  ->apply_network: inp{ts(inp)} target{ts(target)} prd{ts(prd)} targ{ts(targ)}")
							lgm().log(f"\n ** <{self.model_manager.model_name}> E({epoch}/{nepochs})-BATCH[{batch_index}][{tIdx}]: Loss= {sloss.item():.5f}", display=True, end="")
							self.optimizer.zero_grad(set_to_none=True)
							mloss.backward()
							self.optimizer.step()

					self.input[tset.value] = inp
					self.target[tset.value] = targ
					self.product[tset.value] = prd
					ave_loss = losses.item() / ( len(tile_locs) * batch_iter )
					batch_losses.append(ave_loss)

				except Exception as e:
					print( f"\n !!!!! Error processing batch {batch_index} !!!!! {e}")
					print( traceback.format_exc() )

			if self.scheduler is not None:
				self.scheduler.step()

			epoch_time = (time.time() - epoch_start)/60.0
			epoch_loss: float = np.array(batch_losses).mean()
			self.checkpoint_manager.save_checkpoint( epoch, TSet.Train, epoch_loss )
			lgm().log(f'Epoch Execution time: {epoch_time:.1f} min, train-loss: {epoch_loss:.4f}', display=True)
			self.record_eval(epoch)

		train_time = time.time() - train_start
		ntotal_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
		print(f' -------> Training model with {ntotal_params} took {train_time/60:.2f} min.')
		self.current_losses = dict( prediction=epoch_loss, **eval_losses )
		return self.current_losses

	def record_eval(self, epoch: int ):
		for tset in [TSet.Validation, TSet.Test]:
			eval_losses = self.evaluate(tset,epoch=epoch)
			if self.results_accum is not None:
				self.results_accum.record_losses( tset, epoch, eval_losses['validation'], eval_losses['upsampled'] )
			lgm().log(f" ** EVAL {tset.value}, model-loss: {eval_losses['validation']:.4f}, interp-loss: {eval_losses['upsampled']:.4f}", display=True)

	def eval_upscale(self, tset: TSet, **kwargs) -> float:
		seed = kwargs.get('seed', 333)
		torch.manual_seed(seed)
		torch.cuda.manual_seed(seed)
		self.time_index = kwargs.get('time_index', self.time_index)
		self.tile_index = kwargs.get('tile_index', self.tile_index)

		proc_start = time.time()
		tile_locs: Dict[Tuple[int, int], Dict[str, int]] = TileGrid(tset).get_tile_locations(selected_tile=self.tile_index)
		start_coords: List[Union[datetime,int]] = self.input_dataset(TSet.Train).get_batch_start_coords()
		batch_interp_losses = []
		inp,  target, ups = None, None, None
		print( f"start_coords = {start_coords}")
		for batch_index, start_coord in enumerate(start_coords):
			print(f"Processing batch[{batch_index}]: {start_coord}")
			for xyi, tile_loc in tile_locs.items():
				print(f"Processing tile[{xyi}]: {tile_loc}")
				train_data: Dict[str, Tensor] = self.get_srbatch(tile_loc, tset, start_coord)
				inp = train_data['input']
				target: Tensor = train_data['target']
				ups: Tensor = self.upsample(inp)
				batch_interp_losses.append( self.loss(ups, target)[0].item() )
				lgm().log(f" **  ** <{self.model_manager.model_name}:{tset.name}> BATCH[{batch_index}][{xyi}], Loss= {batch_interp_losses[-1]:.5f}", display=True )
		if inp is None: lgm().log( " ---------->> No tiles processed!", display=True)
		self.input[tset.value] = inp
		self.target[tset.value] = target
		self.interp[tset.value] = ups

		proc_time = time.time() - proc_start
		interp_loss: float = np.array(batch_interp_losses).mean()
		ntotal_params: int = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
		lgm().log(f' -------> Exec {tset.value} model with {ntotal_params} wts on {tset.value} tset took {proc_time:.2f} sec, interp loss = {interp_loss:.4f}')
		return interp_loss

	def evaluate(self, tset: TSet, **kwargs) -> Dict[str,float]:
		print( f"  ^^^^^ evaluate ^^^^^ ")
		seed = kwargs.get('seed', 333)
		torch.manual_seed(seed)
		torch.cuda.manual_seed(seed)
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg().task.lr, weight_decay=cfg().task.get('weight_decay', 0.0))
		self.checkpoint_manager = CheckpointManager(self.model, self.optimizer)
		train_state = self.checkpoint_manager.load_checkpoint(tset)
		epoch = kwargs.get( 'epoch', train_state.get('epoch', 0) )
		self.time_index = kwargs.get('time_index', self.time_index)
		self.tile_index = kwargs.get('tile_index', self.tile_index)
		update_checkpoint = kwargs.get('update_checkpoint',True)

		proc_start = time.time()
		tile_locs: Dict[Tuple[int, int], Dict[str, int]] = TileGrid(tset).get_tile_locations(selected_tile=self.tile_index)
		start_coords: List[Union[datetime,int]] = self.input_dataset(tset).get_batch_start_coords()
		batch_model_losses, batch_interp_losses = [], []
		inp, prd, targ, ups, batch_date = None, None, None, None, None
		lgm().log( f"{tset.name} Evaluation: {len(start_coords)} batches", display=True)
		for batch_index, start_coord in enumerate(start_coords):
			for xyi, tile_loc in tile_locs.items():
				train_data: Dict[str, Tensor] = self.get_srbatch(tile_loc, tset, start_coord)
				inp = train_data['input']
				# ups: Tensor = self.get_target_channels(self.upsample(inp))
				ups: Tensor = self.upsample(inp)
				target: Tensor = train_data['target']
				prd, targ = self.apply_network(inp, target)
				batch_model_losses.append( self.loss(prd, targ)[0].item() )
				batch_interp_losses.append( self.loss(ups, targ)[0].item() )
				lgm().log(f" **  ** <{self.model_manager.model_name}:{tset.name}> BATCH[{batch_index}][{xyi}]: inp-mean={inp.mean():.6f}, Loss= {batch_model_losses[-1]:.5f},  Interp-Loss= {batch_interp_losses[-1]:.5f}", display=True )
		if inp is None: lgm().log( " ---------->> No tiles processed!", display=True)
		self.input[tset.value] = inp
		self.target[tset.value] = targ
		self.product[tset.value] = prd

		proc_time = time.time() - proc_start
		model_loss: float = np.array(batch_model_losses).mean()
		interp_loss: float = np.array(batch_interp_losses).mean()
		ntotal_params: int = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
		if (model_loss < self.best_loss[tset]) and update_checkpoint:
			self.checkpoint_manager.save_checkpoint(epoch, tset, model_loss)
			self.best_loss[tset] = model_loss
		lgm().log(f' -------> Exec {tset.value} model with {ntotal_params} wts on {tset.value} tset took {proc_time:.2f} sec, model loss = {model_loss:.4f}, interp loss = {interp_loss:.4f}')
		return dict(validation=model_loss, upsampled=interp_loss)

	def apply_network(self, input_data: Tensor, target_data: Tensor = None ) -> Tuple[TensorOrTensors,Tensor]:
		net_input: Tensor  = input_data
		target: Tensor = target_data
		product: TensorOrTensors = self.model( net_input )
		# if type(product) == torch.Tensor:
		# 	result  =  self.get_target_channels(product)
		# 	#		print( f"get_train_target, input shape={input_data.shape}, product shape={product.shape}, output shape={result.shape}, channel_idxs={channel_idxs}")
		# else:
		# 	result = [ self.get_target_channels(prod) for prod in product ]
		# #		print(f"get_train_target, input shape={input_data.shape}, product shape={product[0].shape}, output shape={result[0].shape}, channel_idxs={channel_idxs}")
		# net_target = self.get_target_channels( target )
		# return result, net_target
		return product, target

	def get_target_channels(self, batch_data: Tensor, tset: TSet) -> Tensor:
		result = tset
		# if self.channel_idxs is None:
		# 	cidxs: List[int] = self.input_dataset(tset).get_channel_idxs(self.target_variables)
		# 	self.channel_idxs = torch.LongTensor( cidxs ).to( self.device )
		# channels = [ torch.select(batch_data, 1, cidx ) for cidx in self.channel_idxs ]
		# result =  channels[0] if (len(channels) == 1) else torch.cat( channels, dim = 1 )
		return result

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

	def apply(self, tile_loc: Dict[str, int], batch_date: datetime, tset: TSet, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
		seed = kwargs.get('seed', 0)
		torch.manual_seed(seed)
		torch.cuda.manual_seed(seed)
		with torch.inference_mode():
			train_data: Dict[str, Tensor] = self.get_srbatch( tile_loc, batch_date, tset )
			inp: Tensor = train_data['input'].squeeze()
			target: Tensor = train_data['target'].squeeze()
			prd,targ = self.apply_network(inp,target)
			loss: torch.Tensor = self.loss(prd, targ)
			filtered_input: np.ndarray = self.model_manager.filter_targets(npa(inp))
			return filtered_input,  npa(targ),  npa(prd), loss.item()
