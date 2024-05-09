import torch, math
import xarray
from torch import Tensor
from typing import Any, Dict, List, Tuple, Type, Optional, Union, Sequence, Mapping, Literal
from fmod.base.util.config import configure, cfg, cfg_date
from fmod.base.util.ops import fmbdir
from fmod.base.util.logging import lgm, exception_handled, log_timing
from fmod.base.util.ops import nnan, pctnan, pctnant, ArrayOrTensor
from fmod.pipeline.merra2 import array2tensor
from enum import Enum
import numpy as np
import torch.nn as nn
import time, os


class CheckpointManager(object):

	def __init__(self):
		self.min_loss = float('inf')

	def save_state(self, model: torch.nn.Module, loss: float ):
		cppath = self.checkpoint_path
		os.makedirs( os.path.dirname(self.checkpoint_path), 0o777, exist_ok=True )
		model_state = model.state_dict()
		torch.save( model_state, cppath )
		if loss < self.min_loss:
			cppath = cppath + ".best"
			lgm().log(f"   ---- Saving best model (loss={loss:.4f}) to {cppath}", display=True )
			model_state['_loss_'] = loss
			torch.save( model_state, cppath )
			self.min_loss = loss

	def load_state(self, model: torch.nn.Module, best_model: bool = False) -> bool:
		cppath = self.checkpoint_path
		if best_model:
			best_cppath = cppath + ".best"
			if os.path.exists( best_cppath ):
				cppath = best_cppath
		if os.path.exists( cppath ):
			try:
				model_state = torch.load( cppath )
				loss = model_state.pop('_loss_', float('inf') )
				model.load_state_dict( model_state )
				self.min_loss = loss
				lgm().log(f"Loaded model from {cppath}, loss = {loss:.2f}", display=True)
				return True
			except Exception as e:
				lgm().log(f"Unable to load model from {cppath}: {e}", display=True)
		return False

	@property
	def checkpoint_path(self) -> str:
		return str(os.path.join(fmbdir('results'), 'checkpoints/' + cfg().model.training_version + ".pt"))

