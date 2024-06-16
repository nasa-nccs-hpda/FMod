import torch
from typing import Any, Dict, List
from fmod.base.util.config import cfg
from fmod.base.util.ops import fmbdir, fmtp
from fmod.base.util.logging import lgm
from torch.optim.optimizer import Optimizer
from torch.nn import Module
from fmod.base.io.loader import TSet
import os


class CheckpointManager(object):

	def __init__(self, model: Module, optimizer: Optimizer ):
		self._cpaths: Dict[str,str] = {}
		self.model = model
		self.optimizer = optimizer

	def save_checkpoint(self, epoch: int, tset: TSet, loss: float ) -> str:
		checkpoint = dict( epoch=epoch, model_state_dict=self.model.state_dict(), optimizer_state_dict=self.optimizer.state_dict(), loss=loss )
		cpath = self.checkpoint_path(tset)
		torch.save( checkpoint, cpath )
		return cpath

	def _load_state(self, tset: TSet ) -> Dict[str,Any]:
		cpath = self.checkpoint_path(tset)
		checkpoint = torch.load(cpath)
		self.model.load_state_dict( checkpoint.pop('model_state_dict') )
		self.optimizer.load_state_dict( checkpoint.pop('optimizer_state_dict') )
		return checkpoint

	def load_checkpoint( self, tset: TSet ) -> Dict[str,Any]:
		cppath = self.checkpoint_path( tset )
		train_state = {}
		if os.path.exists( cppath ):
			try:
				train_state = self._load_state( tset )
				lgm().log(f"Loaded model checkpoint from {cppath}", display=True)
			except Exception as e:
				lgm().log(f"Unable to load model from {cppath}: {e}", display=True)
		else:
			print( f"No checkpoint file found at '{cppath}': starting from scratch.")
		print( f" ------ Saving checkpoints to '{cppath}' ------ " )
		return train_state

	def clear_checkpoint( self, tset: TSet ):
		cppath = self.checkpoint_path(tset)
		if os.path.exists(cppath):
			print( f" >> Clearing state: {cppath}")
			os.remove(cppath)

	def checkpoint_path( self, tset: TSet ) -> str:
		version = tset.value
		if version not in self._cpaths:
			paths = [ f"{fmbdir('results')}/checkpoints/{fmtp('training_version')}.{ext}.pt" for ext in ["current",version] ]
			if os.path.exists(paths[0]):
				self._cpaths[version] = paths[0]
			if os.path.exists(paths[1]) or self._cpaths[version] is None:
				self._cpaths[version] = paths[1]
			os.makedirs(os.path.dirname(self._cpaths[version]), 0o777, exist_ok=True)
		return self._cpaths[version]

