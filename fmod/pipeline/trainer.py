import torch, math
from torch import Tensor
from typing import Any, Dict, List, Tuple, Type, Optional, Union, Sequence, Mapping, Literal
from fmod.base.util.config import configure, cfg, cfg_date
from torch.utils.data import DataLoader
from fmod.base.util.grid import GridOps
import torch_harmonics as harmonics
from fmod.base.io.loader import BaseDataset
from fmod.base.util.ops import fmbdir
from fmod.base.util.logging import lgm, exception_handled, log_timing
import torch.nn as nn
import time, os

class ModelTrainer(object):

	model_cfg = ['batch_size', 'num_workers', 'persistent_workers' ]

	def __init__(self,  dataset: BaseDataset):
		self.dataset = dataset
		self.dataloader = DataLoader( dataset, **self.loader_args )
		inp, tar = next(iter(dataset))
		self.data_iter = iter(dataset)
		self.grid_shape = inp.shape[-2:]
		self.gridops = GridOps(*self.grid_shape)
		lgm().log(f"INPUT={type(inp)}, TARGET={type(tar)}")
		lgm().log(f"SHAPES= {inp.shape}, {tar.shape}, (nlat, nlon)={self.grid_shape}")
		lmax = math.ceil(self.grid_shape[0] / 3)
		self.sht = harmonics.RealSHT( *self.grid_shape, lmax=lmax, mmax=lmax, grid='equiangular', csphase=False)
		self.isht = harmonics.InverseRealSHT( *self.grid_shape, lmax=lmax, mmax=lmax, grid='equiangular', csphase=False)
		self.scheduler = None
		self.optimizer = None
		self.model = None

	def save_state(self):
		os.makedirs( os.path.dirname(self.checkpoint_path), 0o777, exist_ok=True )
		torch.save( self.model.state_dict(), self.checkpoint_path )
		print(f"Saved model to {self.checkpoint_path}")

	def load_state(self) -> bool:
		if os.path.exists( self.checkpoint_path ):
			try:
				self.model.load_state_dict( torch.load( self.checkpoint_path ) )
				print(f"Loaded model from {self.checkpoint_path}")
				return True
			except Exception as e:
				print(f"Unsble to load model from {self.checkpoint_path}: {e}")
		return False

	@property
	def checkpoint_path(self) -> str:
		return str( os.path.join( fmbdir('results'), 'checkpoints/' + cfg().task.training_version) )

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
		load_state = kwargs.get( 'load_state', True )
		save_state = kwargs.get('save_state', True)
		torch.manual_seed(seed)
		torch.cuda.manual_seed(seed)
		self.scheduler = kwargs.get( 'scheduler', None )
		self.model = model
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg().task.lr, weight_decay=cfg().task.weight_decay)
		nepochs = cfg().task.nepochs
		train_start = time.time()
		if load_state: self.load_state()
		for epoch in range(nepochs):
			epoch_start = time.time()
			self.optimizer.zero_grad(set_to_none=True)
			lgm().log(f"\n  ----------- Epoch {epoch + 1}/{nepochs}   ----------- ", display=True )

			acc_loss = 0
			self.model.train()
			for inp, tar in self.data_iter:
				prd = self.model(inp)
				for _ in range( cfg().model.nfuture ):
					prd = self.model(prd)
				if cfg().model.loss_fn == 'l2':
					loss = self.l2loss_sphere( prd, tar)
				elif cfg().model.loss_fn == "spectral l2":
					loss = self.spectral_l2loss_sphere( prd, tar)
				else:
					raise Exception("Unknown loss function {}".format(cfg().model.loss_fn))

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

			# print(f'--------------------------------------------------------------------------------')
			# print(f'Epoch {epoch} summary:')
			# print(f'time taken: {epoch_time}')
			# print(f'accumulated training loss: {acc_loss}')
			# print(f'--------------------------------------------------------------------------------')

			print(f'Epoch {epoch}, time: {epoch_time:.1f}, loss: {acc_loss:.2f}')

		train_time = time.time() - train_start

		print(f'--------------------------------------------------------------------------------')
		print(f'done. Training took {train_time / 60:.2f} min.')
		if save_state: self.save_state()
		return acc_loss

	def inference(self, **kwargs ) -> Tuple[ List[Tensor], List[Tensor], List[Tensor] ]:
		seed = kwargs.get('seed',0)
		max_step = kwargs.get('max_step',5)
		torch.manual_seed(seed)
		torch.cuda.manual_seed(seed)
		inputs, predictions, targets = [], [], []
		with torch.inference_mode():
			for istep, (inp, tar) in enumerate(self.data_iter):
				if istep == max_step: break
				out = self.model(inp).detach()
				predictions.append(out)
				targets.append(tar)
				inputs.append(inp)
		return inputs, targets, predictions