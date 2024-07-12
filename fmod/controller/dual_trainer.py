import torch, math
import xarray, traceback, random
from datetime import datetime
from torch import Tensor
from typing import Any, Dict, List, Tuple, Union, Sequence, Optional
from fmod.base.util.config import ConfigContext, cfg
from fmod.data.tiles import TileIterator
from fmod.base.io.loader import TSet, srRes, batchDomain
from fmod.base.util.config import cdelta, cfg, cval, get_data_coords, dateindex
from fmod.base.gpu import set_device
from fmod.base.util.array import array2tensor, downsample, upsample
from fmod.model.sres.mscnn.network import Upsampler
from fmod.data.batch import BatchDataset
from fmod.data.tiles import TileGrid
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
TimeType = Union[datetime, int]

def ttsplit_times( times: List[TimeType]) -> Dict[TSet, List[TimeType]]:
	ttsplit = cfg().task.ttsplit
	start, result, nt = 0, {}, len(times)
	for tset, tset_fraction in ttsplit.items():
		end = start + int(tset_fraction * nt)
		result[TSet(tset)] = times[start:end]
		start = end
	return result

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

	def __init__(self, cc: ConfigContext ):
		super(ModelTrainer, self).__init__()
		self.model_manager: SRModels = SRModels( set_device() )
		self.device: torch.device = self.model_manager.device
		self.results_accum: ResultsAccumulator = ResultsAccumulator(cc)
		self.domain: batchDomain = batchDomain.from_config(cfg().task.get('batch_domain', 'tiles'))
		self.min_loss = float('inf')
		self.eps = 1e-6
		self._sht, self._isht = None, None
		self.scheduler = None
		self.model = self.model_manager.get_model( )
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg().task.lr, weight_decay=cfg().task.get('weight_decay', 0.0))
		self.checkpoint_manager = CheckpointManager(self.model, self.optimizer)
		self.loss_module: nn.Module = None
		self.layer_losses = []
		self.channel_idxs: torch.LongTensor = None
		self.target_variables = cfg().task.target_variables
		self.downscale_factors = cfg().model.downscale_factors
		self.scale_factor = math.prod(self.downscale_factors)
		self.conform_to_data_grid()
	#	self.grid_shape, self.gridops, self.lmax = self.configure_grid()
		self.input: MLTensors = {}
		self.target: MLTensors = {}
		self.product: MLTensors = {}
		self.interp: MLTensors = {}
		self.upsampled: Tensor = None
		self.current_losses: Dict[str,float] = {}
		self.time_index: int = -1
		self.tile_index: Optional[Tuple[int,int]] = None
		self.validation_loss: float = float('inf')
		self.upsampled_loss: float = float('inf')
		self.data_timestamps: Dict[TSet,List[Union[datetime, int]]] = {}

	@property
	def model_name(self):
		return self.model_manager.model_name

	def get_dataset(self, tset: TSet)-> BatchDataset:
		return self.model_manager.get_dataset( tset )

	def get_sample_input(self, tset: TSet, targets_only: bool = True) -> xa.DataArray:
		return self.model_manager.get_sample_input( tset, targets_only )

	def get_sample_target(self, tset: TSet) -> xa.DataArray:
		return self.model_manager.get_sample_target(tset)


	# def configure_grid(self, tset: TSet ):
	# 	tar: xarray.DataArray = self.target_dataset(tset).get_current_batch_array()
	# 	grid_shape = tar.shape[-2:]
	# 	gridops = GridOps(*grid_shape,self.device)
	# 	lgm().log(f"SHAPES: target{list(tar.shape)}, (nlat, nlon)={grid_shape}")
	# 	lmax = tar.shape[-2]
	# 	return grid_shape, gridops, lmax

	def conform_to_data_grid(self, **kwargs):
		if cfg().task.conform_to_grid:
			data: xarray.DataArray = self.get_dataset(TSet.Train).get_current_batch_array()
			data_origin: Dict[str, float] = get_data_coords(data, cfg().task['origin'])
			dc = cdelta(data)
			lgm().log(f"  ** snap_origin_to_data_grid: {cfg().task['origin']} -> {data_origin}", **kwargs)
			cfg().task['origin'] = data_origin
			cfg().task['extent'] = {dim: float(cval(data, dim, -1) + dc[cfg().task.coords[dim]]) for dim in data_origin.keys()}
			print(f" *** conform_to_data_grid: origin={cfg().task['origin']} extent={cfg().task['extent']} *** ")

	def tensor(self, data: xarray.DataArray) -> torch.Tensor:
		return Tensor(data.values).to(self.device)

	@property
	def loader_args(self) -> Dict[str, Any]:
		return { k: cfg().model.get(k) for k in self.model_cfg }

	def charbonnier(self, prd: torch.Tensor, tar: torch.Tensor) -> torch.Tensor:
		error = torch.sqrt( ((prd - tar) ** 2) + self.eps )
		return torch.mean(error)

	def single_product_loss(self, prd: torch.Tensor, tar: torch.Tensor) -> torch.Tensor:
		if cfg().model.loss_fn == 'l2':
			loss = l2loss(prd, tar)
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

	def get_srbatch(self, ctile: Dict[str,int], ctime: TimeType, tset: TSet,  **kwargs  ) -> Optional[xarray.DataArray]:
		shuffle: bool = kwargs.pop('shuffle',False)
		btarget:  Optional[xarray.DataArray]  = self.get_dataset(tset).get_batch_array(ctile,ctime,**kwargs)
		print( f"get_srbatch({tset.value}): {btarget is not None}")
		if btarget is not None:
			if shuffle:
				batch_perm: Tensor = torch.randperm(btarget.shape[0])
				btarget: xarray.DataArray = btarget[ batch_perm, ... ]
			lgm().log(f" *** target{btarget.dims}{btarget.shape}, mean={btarget.mean():.3f}, std={btarget.std():.3f}")
		return btarget

	def get_ml_input(self, tset: TSet) -> np.ndarray:
		print( f"get_ml_input({tset}): avail={list(self.input.items())}")
		ml_input: Tensor =  self.input[tset]
		result = npa( ml_input ).astype(np.float32)
		return  result

	def get_ml_upsampled(self, tset: TSet) -> np.ndarray:
		return self.upsampled[tset].numpy()

	def get_ml_target(self, tset: TSet) -> np.ndarray:
		return npa( self.target[tset] )

	def get_ml_product(self, tset: TSet) -> np.ndarray:
		return npa(self.product[tset])

	def train(self, **kwargs) -> Dict[str, float]:
		if cfg().task['nepochs'] == 0: return {}
		refresh_state = kwargs.get('refresh_state', False)
		seed = kwargs.get('seed', 4456)

		torch.manual_seed(seed)
		torch.cuda.manual_seed(seed)
		self.scheduler = kwargs.get('scheduler', None)
		epoch0, epoch_loss, nepochs, loss_history, eval_losses, tset = 0, 0.0, cfg().task.nepochs,  [], {}, TSet.Train
		train_start = time.time()
		if refresh_state:
			self.checkpoint_manager.clear_checkpoints()
			if self.results_accum is not None:
				self.results_accum.refresh_state()
			print(" *** No checkpoint loaded: training from scratch *** ")
		else:
			train_state = self.checkpoint_manager.load_checkpoint( update_model=True )
			if self.results_accum is not None:
				self.results_accum.load_results()
			epoch0 = train_state.get('epoch', 0)
			epoch_loss = train_state.get('loss', float('inf'))
			nepochs += epoch0

		self.init_data_timestamps()
		for epoch in range(epoch0+1,nepochs+1):
			epoch_start = time.time()
			self.optimizer.zero_grad(set_to_none=True)
			self.model.train()

			ctiles = TileIterator(TSet.Train)
			lgm().log(f"  ----------- Epoch {epoch}/{nepochs}   ----------- ", display=True )

			batch_losses = torch.tensor(0.0, device=self.device, dtype=torch.float32)
			binput, boutput, btarget, ibatch = None, None, None, 0
			for itime, ctime in enumerate(self.data_timestamps[TSet.Train]):
				for ctile in iter(ctiles):
					batch_data: Optional[xa.DataArray] = self.get_srbatch(ctile,ctime,TSet.Train)
					if batch_data is None: break
					binput, boutput, btarget = self.apply_network( batch_data )
					lgm().log(f"  ->apply_network: inp{binput.shape} target{ts(btarget)} prd{ts(boutput)}" )
					[sloss, mloss] = self.loss(boutput,btarget)
					lgm().log(f"\n ** <{self.model_manager.model_name}> E({epoch}/{nepochs})-BATCH[{itime}][{ibatch}]: Loss= {sloss.item():.5f}", display=True, end="")
					self.optimizer.zero_grad(set_to_none=True)
					mloss.backward()
					self.optimizer.step()
					batch_losses += sloss
					ibatch += 1

			if binput is not None:   self.input[tset] = binput
			if btarget is not None:  self.target[tset] = btarget
			if boutput is not None:  self.product[tset] = boutput

			if self.scheduler is not None:
				self.scheduler.step()

			epoch_loss: float = batch_losses.item() / ibatch
			epoch_time = (time.time() - epoch_start)/60.0
			self.checkpoint_manager.save_checkpoint( epoch, TSet.Train, epoch_loss )
			lgm().log(f'Epoch Execution time: {epoch_time:.1f} min, train-loss: {epoch_loss:.4f}', display=True)
			self.record_eval( epoch, {TSet.Train: epoch_loss}, TSet.Validation )

		train_time = time.time() - train_start
		ntotal_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
		self.record_eval( nepochs, {},  TSet.Test )
		print(f'\n -------> Training model with {ntotal_params} wts took {train_time/60:.2f} ({train_time/(60*nepochs):.2f} per epoch) min.')
		self.current_losses = dict( prediction=epoch_loss, **eval_losses )
		return self.current_losses

	def record_eval(self, epoch: int, losses: Dict[TSet,float], tset: TSet, **kwargs ):
		eval_losses = self.evaluate( tset, upsample=(epoch==0), **kwargs )
		if self.results_accum is not None:
			print( f" --->> record {tset.name} eval[{epoch}]: eval_losses={eval_losses}, losses={losses}")
			self.results_accum.record_losses( tset, epoch, eval_losses['model'] )
			if 'upsample' in eval_losses:
				self.results_accum.record_losses( TSet.Upsample, epoch, eval_losses['upsample'])
			for etset, loss in losses.items():
				self.results_accum.record_losses( etset, epoch, loss )
		if kwargs.get('flush',True):
			self.results_accum.flush()
		return eval_losses

	def init_data_timestamps(self, tset: TSet):
		if len(self.data_timestamps) == 0:
			ctimes: List[TimeType] = self.get_dataset(tset).get_batch_time_coords()
			lgm().log( f"init_data_timestamps[{tset.value}]: {ctimes}", display=True)
			self.data_timestamps = ttsplit_times(ctimes)


	def evaluate(self, tset: TSet, **kwargs) -> Dict[str,float]:
		seed = kwargs.get('seed', 333)
		interp_loss = kwargs.get('interp_loss', False)
		assert tset in [ TSet.Validation, TSet.Test ], f"Invalid tset in training evaluation: {tset.name}"
		torch.manual_seed(seed)
		torch.cuda.manual_seed(seed)
		self.time_index = kwargs.get('time_index', self.time_index)
		self.tile_index = kwargs.get('tile_index', self.tile_index)
		train_state = self.checkpoint_manager.load_checkpoint( TSet.Validation, **kwargs )
		self.validation_loss = train_state.get('loss', float('inf'))
		epoch = train_state.get( 'epoch', 0 )
		self.init_data_timestamps(tset)

		proc_start = time.time()
		ctiles = TileIterator(TSet.Train)
		ctimes = self.data_timestamps[tset]

		batch_model_losses, batch_interp_losses = [], []
		binput, boutput, btarget, ibatch = None, None, None, 0
		lgm().log( f"evaluate[{tset.name}]: ctimes={ctimes}, ctiles={ctiles}",display=True)
		for itime, ctime in enumerate(ctimes):
			print( f" @Evaluating ctime[{itime}]: {ctime}")
			for ctile in iter(ctiles):
				batch_data: Optional[xa.DataArray] = self.get_srbatch(ctile, ctime, tset)
				if batch_data is None: break
				binput, boutput, btarget = self.apply_network( batch_data )
				lgm().log(f"  ->apply_network: inp{ts(binput)} target{ts(btarget)} prd{ts(boutput)}" )
				[model_sloss, model_multilevel_loss] = self.loss(boutput, btarget)
				batch_model_losses.append( model_sloss.item() )
				if interp_loss:
					binterp = upsample(binput)
					[interp_sloss, interp_multilevel_mloss] = self.loss(boutput, binterp)
					batch_interp_losses.append( interp_sloss.item() )
				lgm().log(f" **  ** <{self.model_manager.model_name}:{tset.name}> BATCH[{ibatch}]: Loss= {batch_model_losses[-1]:.5f}", display=True )
				ibatch = ibatch + 1
		if binput is not None:  self.input[tset] = binput
		if btarget is not None: self.target[tset] = btarget
		if boutput is not None: self.product[tset] = boutput

		proc_time = time.time() - proc_start
		model_loss: float = np.array(batch_model_losses).mean()
		ntotal_params: int = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
		if (tset == TSet.Validation) and (model_loss < self.validation_loss):
			self.validation_loss = model_loss
			self.checkpoint_manager.save_checkpoint( epoch, TSet.Validation, self.validation_loss )
		lgm().log(f' -------> Exec {tset.value} model with {ntotal_params} wts on {tset.value} tset took {proc_time:.2f} sec, model loss = {model_loss:.4f}')
		result = dict( model=model_loss )
		if interp_loss:
			result['upsample'] = np.array(batch_interp_losses).mean()
		return result

	def apply_network(self, target_data: xa.DataArray ) -> Tuple[Tensor,TensorOrTensors,Tensor]:
		target_channels = cfg().task.target_variables
		btarget: Tensor = array2tensor( target_data.sel(channels=target_channels) )
		binput = downsample(target_data)
		product: TensorOrTensors = self.model( binput )
		return binput, product, btarget

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

