import torch
from typing import Any, Dict
from fmod.base.util.config import cfg
from fmod.base.util.ops import fmbdir, fmtp
from fmod.base.util.logging import lgm
from torch.optim.optimizer import Optimizer
from torch.nn import Module
import os


class CheckpointManager(object):

	def __init__(self, model: Module, optimizer: Optimizer ):
		self.min_loss = float('inf')
		self._cpaths: Dict[str,str] = {}
		self.model = model
		self.optimizer = optimizer

	def _save_state(self, epoch: int, loss: float, version: str = "current" ) -> str:
		checkpoint = dict( epoch=epoch, model_state_dict=self.model.state_dict(), optimizer_state_dict=self.optimizer.state_dict(), loss=loss )
		cpath = self.checkpoint_path(version)
		torch.save( checkpoint, cpath )
		return cpath

	def save_checkpoint(self, epoch: int, loss: float ):
		cpath = self._save_state( epoch, loss )
		lgm().log(f" -- Checkpoint saved --", display=True)
		if loss < self.min_loss:
			cppath = self._save_state( epoch, loss, "best" )
			lgm().log(f"   ---- Saving best model (loss={loss:.4f})", display=True )
			self.min_loss = loss

	def _load_state(self, version: str ) -> Dict[str,Any]:
		cpath = self.checkpoint_path(version)
		checkpoint = torch.load(cpath)
		self.model.load_state_dict( checkpoint.pop('model_state_dict') )
		self.optimizer.load_state_dict( checkpoint.pop('optimizer_state_dict') )
		return checkpoint

	def load_checkpoint( self, version: str ) -> Dict[str,Any]:
		cppath = self.checkpoint_path( version )
		train_state = {}
		if os.path.exists( cppath ):
			try:
				train_state = self._load_state( version)
				self.min_loss = train_state.get('loss',float('inf'))
				lgm().log(f"Loaded model from {cppath}, loss = {self.min_loss:.2f}", display=True)
			except Exception as e:
				lgm().log(f"Unable to load model from {cppath}: {e}", display=True)
		else:
			print( f"No checkpoint file found at '{cppath}': starting from scratch.")
		print( f" ------ Saving checkpoints to '{self.checkpoint_path()}' ------ " )
		return train_state

	def checkpoint_path(self, version="current") -> str:
		if version not in self._cpaths:
			self._cpaths[version] =  str(os.path.join(fmbdir('results'), 'checkpoints/' + fmtp('training_version') + f".{version}.pt"))
			os.makedirs(os.path.dirname(self._cpaths[version]), 0o777, exist_ok=True)
		return self._cpaths[version]

