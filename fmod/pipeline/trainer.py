import torch, math
import xarray
from torch import Tensor
from typing import Any, Dict, List, Tuple, Type, Optional, Union, Sequence, Mapping, Literal
from fmod.base.util.config import cfg, cfg_date
from fmod.pipeline.downscale import Downscaler
from fmod.base.util.grid import GridOps
import torch_harmonics as harmonics
from fmod.base.io.loader import BaseDataset
from fmod.base.util.ops import fmbdir
from fmod.base.util.logging import lgm, exception_handled, log_timing
from fmod.base.util.ops import nnan, pctnan, pctnant, ArrayOrTensor
from fmod.pipeline.merra2 import array2tensor
from fmod.pipeline.checkpoints import CheckpointManager
from enum import Enum
import numpy as np
import torch.nn as nn
import time, os

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
	return tensor.detach().cpu().numpy().squeeze()
class TaskType(Enum):
	Downscale = 'downscale'
	Forecast = 'forecast'

class ModelTrainer(object):

	model_cfg = ['batch_size', 'num_workers', 'persistent_workers' ]

	def __init__(self,  dataset: BaseDataset, device: torch.device ):
		self.dataset = dataset
		self.device = device
		self.scale_factor = cfg().model.get('scale_factor',1)
		self.task_type: TaskType = TaskType(cfg().task.task_type)
		inp, tar = next(iter(dataset))
		self.data_iter = iter(dataset)
		if self.task_type == TaskType.Downscale:
			self.grid_shape = tar.shape[-2:]
			lmax = inp.shape[-2]
		else:
			self.grid_shape = inp.shape[-2:]
			lmax = math.ceil(self.grid_shape[0] / cfg().model.get('scale_factor',1))
		self.gridops = GridOps(*self.grid_shape)
		lgm().log(f"SHAPES: input{list(inp.shape)}, target{list(tar.shape)}, (nlat, nlon)={self.grid_shape}, lmax={lmax}", display=True)
	#	self.sht = harmonics.RealSHT( *self.grid_shape, lmax=lmax, mmax=lmax, grid='equiangular', csphase=False)
	#	self.isht = harmonics.InverseRealSHT( *self.grid_shape, lmax=lmax, mmax=lmax, grid='equiangular', csphase=False)
		self.scheduler = None
		self.optimizer = None
		self.model = None
		self.checkpoint_manager = None

	def tensor(self, data: xarray.DataArray) -> torch.Tensor:
		return Tensor(data.values).to(self.device)

	def save_state(self):
		os.makedirs( os.path.dirname(self.checkpoint_path), 0o777, exist_ok=True )
		torch.save( self.model, self.checkpoint_path )

	def load_state(self) -> bool:
		if os.path.exists( self.checkpoint_path ):
			try:
				self.model = torch.load( self.checkpoint_path )
				lgm().log(f"Loaded model from {self.checkpoint_path}", display=True)
				return True
			except Exception as e:
				lgm().log(f"Unable to load model from {self.checkpoint_path}: {e}", display=True)
		return False

	@property
	def checkpoint_path(self) -> str:
		return str( os.path.join( fmbdir('results'), 'checkpoints/' + cfg().model.training_version + ".pt") )

	@property
	def loader_args(self) -> Dict[str, Any]:
		return { k: cfg().model.get(k) for k in self.model_cfg }

	def l2loss_sphere(self, prd, tar, relative=False, squared=True):
		loss = self.gridops.integrate_grid((prd - tar) ** 2, dimensionless=True).sum(dim=-1)
		if relative:
			loss = loss / self.gridops.integrate_grid(tar ** 2, dimensionless=True).sum(dim=-1)

		if not squared:
			loss = torch.sqrt(loss)
		loss = loss.mean()

		return loss

	def spectral_l2loss_sphere(self, prd, tar, relative=False, squared=True):
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

	@exception_handled
	def train(self, model: nn.Module, **kwargs ):
		seed = kwargs.get('seed',333)
		load_state = kwargs.get( 'load_state', '' )
		save_state = kwargs.get('save_state', True)
		torch.manual_seed(seed)
		torch.cuda.manual_seed(seed)
		self.scheduler = kwargs.get( 'scheduler', None )
		self.model = model
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg().task.lr, weight_decay=cfg().task.weight_decay)
		self.checkpoint_manager = CheckpointManager(self.model,self.optimizer)
		epoch0, nepochs = 0, cfg().task.nepochs
		train_start = time.time()
		if load_state:
			train_state = self.checkpoint_manager.load_checkpoint(load_state)
			epoch0 = train_state['epoch']
			nepochs += epoch0

		for epoch in range(nepochs):
			epoch_start = time.time()
			self.optimizer.zero_grad(set_to_none=True)
			lgm().log(f"\n  ----------- Epoch {epoch + 1}/{nepochs}   ----------- " )

			acc_loss = 0
			self.model.train()
			for inp, tar in self.data_iter:
				prd = self.model( inp )
				for _ in range( cfg().model.nfuture ):
					prd = self.model( prd )
				if cfg().model.loss_fn == 'l2':
					loss = self.l2loss_sphere( prd, tar )
				elif cfg().model.loss_fn == "spectral-l2":
					loss = self.spectral_l2loss_sphere( prd, tar )
				else:
					raise Exception("Unknown loss function {}".format(cfg().model.loss_fn))
				lgm().log(f"\n  ----------- Epoch {epoch + 1}/{nepochs}   ----------- ")
				lgm().log(f" ** inp shape={list(inp.shape)}, pct-nan= {pctnant(inp)}")
				lgm().log(f" ** tar shape={list(tar.shape)}, pct-nan= {pctnant(tar)}")
				lgm().log(f" ** prd shape={list(prd.shape)}, pct-nan= {pctnant(prd)}")

				acc_loss += loss.item() * inp.size(0)
				#        print( f"Loss: {loss.item()}")

				self.optimizer.zero_grad(set_to_none=True)
				# gscaler.scale(loss).backward()
				loss.backward()
				self.optimizer.step()
			# gscaler.update()

			if self.scheduler is not None:
				self.scheduler.step()

			acc_loss = acc_loss / len(self.dataset)
			epoch_time = time.time() - epoch_start

			cp_msg = ""
			if save_state:
				self.checkpoint_manager.save_checkpoint( epoch, acc_loss )
				cp_msg = "  ** model saved ** "
			lgm().log(f'Epoch {epoch}, time: {epoch_time:.1f}, loss: {acc_loss:.2f}  {cp_msg}', display=True)

		train_time = time.time() - train_start

		print(f'--------------------------------------------------------------------------------')
		print(f'done. Training took {train_time / 60:.2f} min.')

		return acc_loss

	def inference(self, **kwargs ) -> Tuple[ List[np.ndarray], List[np.ndarray], List[np.ndarray] ]:
		seed = kwargs.get('seed',0)
		max_step = kwargs.get('max_step',5)
		torch.manual_seed(seed)
		torch.cuda.manual_seed(seed)
		inputs, predictions, targets = [], [], []
		with torch.inference_mode():
			for istep, (inp, tar) in enumerate(self.data_iter):
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


class DualModelTrainer(object):

	model_cfg = ['batch_size', 'num_workers', 'persistent_workers' ]

	def __init__(self,  input_dataset: BaseDataset,  target_dataset: BaseDataset, device: str = "cpu", batch_size=1 ):
		self.device = device
		self.batch_size = batch_size
		self.input_dataset = input_dataset
		self.target_dataset = target_dataset
		self.chanids = input_dataset.channel_ids
		self.scale_factor = cfg().model.get('scale_factor',1)
		self.task_type: TaskType = TaskType(cfg().task.task_type)
		self.scale_factor = cfg().model.get('scale_factor',1)
		sample_input = next(iter(input_dataset))
		sample_target = next(iter(target_dataset))
		for k,v in sample_input.items():
			print( f" * input[{k}]: {v.dims} {v.shape}" )
		for k, v in sample_target.items():
			print( f" * target[{k}]: {v.dims} {v.shape}" )
		self.input_grid = sample_input['input'].shape[-2:]
		self.output_grid = sample_target['target'].shape[-2:]
		self.input_data_iter = iter(input_dataset)
		self.target_data_iter = iter(target_dataset)
		lmax = math.ceil(self.output_grid[0] / cfg().model.get('scale_factor',1))
		self.gridops = GridOps(*self.output_grid)
		self.sht = harmonics.RealSHT( *self.output_grid, lmax=lmax, mmax=lmax, grid='equiangular', csphase=False)
		self.isht = harmonics.InverseRealSHT( *self.output_grid, lmax=lmax, mmax=lmax, grid='equiangular', csphase=False)
		self.scheduler = None
		self.optimizer = None
		self.model = None
		self.checkpoint_manager = None
		self.downscaler = Downscaler()

	def __iter__(self):
		self.input_data_iter = iter(self.input_dataset)
		self.target_data_iter = iter(self.target_dataset)
		return self

	def __next__(self) -> Tuple[xarray.DataArray, xarray.DataArray]:
		print( f"DualModelTrainer: next")
		inputs, targets = [], []
		for iB in range(self.batch_size):
			inp = next(self.input_data_iter)
			tar = next(self.target_data_iter)
			inputs.append( inp['input'] )
			targets.append( tar['target'] )
		return batch(inputs), batch(targets)

	def tensor(self, data: xarray.DataArray) -> torch.Tensor:
		return Tensor(data.values).to(self.device)

	@property
	def loader_args(self) -> Dict[str, Any]:
		return { k: cfg().model.get(k) for k in self.model_cfg }

	def l2loss_sphere(self, prd, tar, relative=False, squared=True):
		loss = self.gridops.integrate_grid((prd - tar) ** 2, dimensionless=True).sum(dim=-1)
		if relative:
			loss = loss / self.gridops.integrate_grid(tar ** 2, dimensionless=True).sum(dim=-1)

		if not squared:
			loss = torch.sqrt(loss)
		loss = loss.mean()

		return loss

	def spectral_l2loss_sphere(self, prd, tar, relative=False, squared=True):
		# compute coefficients
		coeffs = torch.view_as_real(self.model.sht(prd - tar))
		coeffs = coeffs[..., 0] ** 2 + coeffs[..., 1] ** 2
		norm2 = coeffs[..., :, 0] + 2 * torch.sum(coeffs[..., :, 1:], dim=-1)
		loss = torch.sum(norm2, dim=(-1, -2))

		if relative:
			tar_coeffs = torch.view_as_real(self.model.sht(tar))
			tar_coeffs = tar_coeffs[..., 0] ** 2 + tar_coeffs[..., 1] ** 2
			tar_norm2 = tar_coeffs[..., :, 0] + 2 * torch.sum(tar_coeffs[..., :, 1:], dim=-1)
			tar_norm2 = torch.sum(tar_norm2, dim=(-1, -2))
			loss = loss / tar_norm2

		if not squared:
			loss = torch.sqrt(loss)
		loss = loss.mean()

		return loss

	@exception_handled
	def train(self, model: nn.Module, **kwargs ):
		print( f" *------> training: {kwargs}" )
		seed = kwargs.get('seed',333)
		load_state = kwargs.get( 'load_state', '' )
		save_state = kwargs.get('save_state', True)
		torch.manual_seed(seed)
		torch.cuda.manual_seed(seed)
		self.scheduler = kwargs.get( 'scheduler', None )
		self.model = model
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg().task.lr, weight_decay=cfg().task.weight_decay)
		self.checkpoint_manager = CheckpointManager(self.model,self.optimizer)
		epoch0, nepochs = 0, cfg().task.nepochs
		train_start = time.time()
		if load_state:
			train_state = self.checkpoint_manager.load_checkpoint(load_state)
			epoch0 = train_state['epoch']
			nepochs += epoch0

		for epoch in range(epoch0,nepochs):
			print(f'Epoch {epoch + 1}/{nepochs}: ')
			epoch_start = time.time()
			self.optimizer.zero_grad(set_to_none=True)
			acc_loss, acc_base_loss = 0, 0
			self.model.train()
			for iT, (xinp, xtar) in enumerate(iter(self)):
				print(f' *** STEP-{iT}, input{xinp.dims}{xinp.shape}, target{xtar.dims}{xtar.shape}')
				inp = array2tensor(xinp)
				tar = array2tensor(xtar)
				prd = self.model(inp)
				for _ in range( cfg().model.nfuture ):
					prd = self.model(prd)
				prd = prd.squeeze()
				if cfg().model.loss_fn == 'l2':
					loss = self.l2loss_sphere( prd, tar )
				elif cfg().model.loss_fn == "spectral-l2":
					loss = self.spectral_l2loss_sphere( prd, tar)
				else:
					raise Exception("Unknown loss function {}".format(cfg().model.loss_fn))

				current_loss = loss.item()
				acc_loss += current_loss

				lgm().log(f" * E-{epoch+1} T-{iT}, loss: {current_loss:.2f} -> prd{list(prd.shape)} - tar{list(tar.shape)}" )
				self.optimizer.zero_grad(set_to_none=True)
				# gscaler.scale(loss).backward()
				loss.backward()
				self.optimizer.step()
			# gscaler.update()

			if self.scheduler is not None:
				self.scheduler.step()

			acc_loss = acc_loss / len(self.input_dataset)
			epoch_time = time.time() - epoch_start
			cp_msg = ""
			if save_state:
				self.checkpoint_manager.save_checkpoint(epoch,acc_loss)
				cp_msg = "  ** model saved ** "
			lgm().log(f' ---------- Epoch {epoch+1}, time: {epoch_time:.1f}, loss: {acc_loss:.2f} {cp_msg}', display=True)

		train_time = time.time() - train_start
		print(f'--------------------------------------------------------------------------------')
		print(f'done. Training took {train_time / 60:.2f} min.')

		return acc_loss

	def l2s_error(self, prd, tar ) -> torch.Tensor:
		loss = self.gridops.integrate_grid((prd - tar) ** 2, dimensionless=True)
		loss = torch.sqrt(loss)
		return loss

	def spectral_l2s_error(self, prd: Tensor, tar: Tensor ) -> torch.Tensor:
		coeffs = torch.view_as_real(self.sht(prd - tar))
		coeffs = coeffs[..., 0] ** 2 + coeffs[..., 1] ** 2
		norm2 = coeffs[..., :, 0] + 2 * torch.sum(coeffs[..., :, 1:], dim=-1)
		loss = torch.sum(norm2, dim=(-1, -2))
		loss = torch.sqrt(loss)
		return loss

	def error(self, etype: str, data: xarray.DataArray, target: xarray.DataArray, dims: List[str]) -> np.ndarray:
		if etype == "mse" :
			sdiff: xarray.DataArray = (target - data) ** 2
			return np.sqrt( sdiff.mean(dim=dims).values )
		elif etype == 'l2':
			error = self.l2s_error( array2tensor(data), array2tensor(target) )
			return error.detach().cpu().numpy()
		elif etype == "spectral-l2":
			error = self.spectral_l2s_error( array2tensor(data), array2tensor(target) )
			return  error.detach().cpu().numpy()
		else:
			raise Exception(f"Unknown error function {etype}")

	def inference(self, **kwargs ) -> Tuple[ List[xarray.DataArray], List[xarray.DataArray], List[xarray.DataArray], List[xarray.DataArray] ]:
		seed = kwargs.get('seed',0)
		max_step = kwargs.get( 'max_step', -1)
		torch.manual_seed(seed)
		torch.cuda.manual_seed(seed)
		inputs, predictions, targets = [], [], []
		interpolates: List[xarray.DataArray] = []
		acc_loss, acc_interp_loss = 0, 0
		with torch.inference_mode():
			for istep, (xinp, xtar, xbase) in enumerate( iter(self) ):
				if istep == max_step: break
				out: Tensor = self.model( array2tensor(xinp) )
				tar: Tensor = array2tensor(xtar)
				prediction: xarray.DataArray = xtar.copy( data=npa(out) )
				predictions.append( prediction )
				targets.append( xtar )
				inputs.append( xinp )
				interpolate: xarray.DataArray = self.downscaler.interpolate(xbase, xtar)
				interpolates.append( interpolate )
				if cfg().model.loss_fn == 'l2':
					loss = self.l2loss_sphere( out, tar )
					interp_loss = self.l2loss_sphere( array2tensor(interpolate), tar )
				elif cfg().model.loss_fn == "spectral-l2":
					loss = self.spectral_l2loss_sphere( out, tar )
					interp_loss = self.spectral_l2loss_sphere( array2tensor(interpolate), tar )
				else:
					raise Exception("Unknown loss function {}".format(cfg().model.loss_fn))
				lgm().log(f' * STEP {istep}: in{xinp.dims}{list(xinp.shape)}, prediction{prediction.dims}{list(prediction.shape)}, tar{xtar.dims}{list(xtar.shape)}, inter{interpolate.dims}{list(interpolate.shape)}, loss={loss:.2f}, interp_loss={interp_loss:.2f}' )
				if istep == 0:
					lgm().log( f"\n----  STATS COMP:  prediction <-> interpolation  ---- ", display=True )
					self.error_comp( kwargs.get('etype','mse'), prediction, interpolate, xtar, ["lat", "lon"], cids=self.chanids('target'), display=True )

				acc_interp_loss += interp_loss.item()
				acc_loss += loss.item()

		acc_loss = acc_loss / len(self.input_dataset)
		acc_interp_loss = acc_interp_loss / len(self.input_dataset)
		lgm().log(f" ** Accumulated Loss: {acc_loss}, Accum Interp Loss: {acc_interp_loss}", display=True)


		return inputs, targets, predictions, interpolates

	def error_comp(self, etype: str, result1: xarray.DataArray, result2: xarray.DataArray, target: xarray.DataArray, dims: List[str], **kwargs):
		loss1 = self.error( etype, result1, target, dims).tolist()
		loss2 = self.error( etype, result2, target, dims).tolist()
		cids: List[str] = kwargs.get('cids', [])
		for iC, (l1, l2) in enumerate(zip(loss1, loss2)):
			cid = cids[iC] if len(cids) else f"C-{iC}"
			lgm().log(f" *{cid:<25}: ---  loss[ {l1:6.3f}, {l2:6.3f} ]", display=kwargs.get('display', False))
