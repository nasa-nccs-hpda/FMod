import torch, math
import xarray
from datetime import date
from torch import Tensor
from typing import Any, Dict, List, Tuple, Type, Optional, Union, Sequence, Mapping, Literal
from fmod.base.util.config import configure, cfg, cfg_date
from fmod.base.util.grid import GridOps
from fmod.pipeline.merra2 import array2tensor
import torch_harmonics as harmonics
from fmod.base.io.loader import BaseDataset
from fmod.base.util.ops import fmbdir
from fmod.base.util.logging import lgm, exception_handled, log_timing
from fmod.base.util.ops import nnan, pctnan, pctnant, ArrayOrTensor
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


class ModelTrainer(object):

	model_cfg = ['batch_size', 'num_workers', 'persistent_workers' ]

	def __init__(self,  input_dataset: BaseDataset, target_dataset: BaseDataset, device: torch.device ):
		self.input_dataset = input_dataset
		self.target_dataset = target_dataset
		self.device = device
		self.scale_factor = cfg().model.get('scale_factor',1)
		self.min_loss = float('inf')
		sample_results = next(iter(target_dataset))
		tar: xarray.DataArray = sample_results['target']
		self.grid_shape = tar.shape[-2:]
		self.gridops = GridOps(*self.grid_shape)
		lgm().log(f"SHAPES: target{list(tar.shape)}, (nlat, nlon)={self.grid_shape}", display=True)
		self.lmax = tar.shape[-2]
		self._sht, self._isht = None, None
		self.scheduler = None
		self.optimizer = None
		self.model = None

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

	def save_state(self, loss: float ):
		cppath = self.checkpoint_path
		os.makedirs( os.path.dirname(self.checkpoint_path), 0o777, exist_ok=True )
		model_state = self.model.state_dict()
		torch.save( model_state, cppath )
		if loss < self.min_loss:
			cppath = cppath + ".best"
			lgm().log(f"   ---- Saving best model (loss={loss:.2f}) to {cppath}", display=True )
			model_state['_loss_'] = loss
			torch.save( model_state, cppath )
			self.min_loss = loss

	def load_state(self, best_model: bool = False) -> bool:
		cppath = self.checkpoint_path
		if best_model:
			best_cppath = cppath + ".best"
			if os.path.exists( best_cppath ):
				cppath = best_cppath
		if os.path.exists( cppath ):
			try:
				model_state = torch.load( cppath )
				loss = model_state.pop('_loss_', float('inf') )
				self.model.load_state_dict( model_state )
				self.min_loss = loss
				lgm().log(f"Loaded model from {cppath}, loss = {loss:.2f}", display=True)
				return True
			except Exception as e:
				lgm().log(f"Unable to load model from {cppath}: {e}", display=True)
		return False

	@property
	def checkpoint_path(self) -> str:
		return str( os.path.join( fmbdir('results'), 'checkpoints/' + cfg().model.training_version) )

	@property
	def loader_args(self) -> Dict[str, Any]:
		return { k: cfg().model.get(k) for k in self.model_cfg }

	def l2loss(self, prd: torch.Tensor, tar: torch.Tensor, squared=True) -> torch.Tensor:
		loss = ((prd - tar) ** 2).sum()
		if not squared:
			loss = torch.sqrt(loss)
		loss = loss.mean()
		return loss

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

	def loss(self, prd: torch.Tensor, tar: torch.Tensor) -> torch.Tensor:
		if cfg().model.loss_fn == 'l2':
			loss = self.l2loss(prd, tar)
		elif cfg().model.loss_fn == 'l2s':
			loss = self.l2loss_sphere(prd, tar)
		elif cfg().model.loss_fn == "spectral-l2s":
			loss = self.spectral_l2loss_sphere(prd, tar)
		else:
			raise Exception("Unknown loss function {}".format(cfg().model.loss_fn))
		return loss

	def get_batch(self, batch_date, as_tensor: bool = True ) -> Dict[str,Union[torch.Tensor,xarray.DataArray]]:
		input_batch: Dict[str, xarray.DataArray]  = self.input_dataset.get_batch(batch_date)
		target_batch: Dict[str, xarray.DataArray] = self.target_dataset.get_batch(batch_date)
		binput: xarray.DataArray = input_batch['input']
		btarget: xarray.DataArray = target_batch['target']
		lgm().log(f" *** input{binput.dims}{binput.shape}, pct-nan= {pctnan(binput.values)}")
		lgm().log(f" *** target{btarget.dims}{btarget.shape}, pct-nan= {pctnan(btarget.values)}")
		if as_tensor:  return dict( input=array2tensor(binput), target=array2tensor(btarget) )
		else:          return dict( input=binput,               target=btarget )

	@exception_handled
	def train(self, model: nn.Module, **kwargs ):
		seed = kwargs.get('seed',333)
		load_state = kwargs.get( 'load_state', True )
		best_model = kwargs.get('best_model', True)
		save_state = kwargs.get('save_state', True)

		torch.manual_seed(seed)
		torch.cuda.manual_seed(seed)
		self.scheduler = kwargs.get( 'scheduler', None )
		self.model = model
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg().task.lr, weight_decay=cfg().task.weight_decay)
		nepochs = cfg().task.nepochs
		train_start, acc_loss = time.time(), 0
		if load_state: self.load_state(best_model)
		for epoch in range(nepochs):
			epoch_start = time.time()
			self.optimizer.zero_grad(set_to_none=True)
			lgm().log(f"  ----------- Epoch {epoch + 1}/{nepochs}   ----------- ", display=True )

			acc_loss: float = 0
			self.model.train()
			batch_dates: List[date] = self.input_dataset.randomize()
			for batch_date in batch_dates:
				train_data: Dict[str,torch.Tensor] = self.get_batch(batch_date)
				input: torch.Tensor = train_data['input']
				target: torch.Tensor   = train_data['target']
				prd:  torch.Tensor  = self.model( input )
				lgm().log( f" LOSS shapes: input={list(input.shape)}, target={list(target.shape)}, product={list(prd.shape)}")
				loss: torch.Tensor  = self.loss( prd, target )
				acc_loss += loss.item() * train_data['input'].size(0)
				lgm().log(f" ** Loss[{batch_date}]: {loss.item():.2f}")

				self.optimizer.zero_grad(set_to_none=True)
				loss.backward()
				self.optimizer.step()

			if self.scheduler is not None:
				self.scheduler.step()

			acc_loss = acc_loss / len(self.input_dataset)
			epoch_time = time.time() - epoch_start

			cp_msg = ""
			if save_state:
				self.save_state( acc_loss )
				cp_msg = "  ** model saved ** "
			lgm().log(f'Epoch {epoch}, time: {epoch_time:.1f}, loss: {acc_loss:.2f}  {cp_msg}', display=True)

		train_time = time.time() - train_start

		print(f'--------------------------------------------------------------------------------')
		print(f'done. Training took {train_time / 60:.2f} min.')

		return acc_loss

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

	def apply(self, date_index: int, **kwargs ) -> Tuple[ np.ndarray, np.ndarray, np.ndarray ]:
		seed = kwargs.get('seed',0)
		torch.manual_seed(seed)
		torch.cuda.manual_seed(seed)
		with torch.inference_mode():
			input_batch = self.input_dataset[date_index]
			target_batch = self.target_dataset[date_index]
			inp: torch.Tensor = array2tensor( input_batch['input'] )
			tar: torch.Tensor = array2tensor( target_batch['target'] )
			out: Tensor = self.model(inp)
			lgm().log(f' * in: {list(inp.shape)}, target: {list(tar.shape)}, out: {list(out.shape)}', display=True)
			return npa(inp), npa(tar), npa(out)
