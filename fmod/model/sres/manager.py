import logging, torch, math
from fmod.base.util.logging import lgm, exception_handled, log_timing
import torch.nn as nn
import xarray as xa
import os, time
from fmod.base.util.config import cfg
from typing import Any, Dict, List, Tuple, Type, Optional, Union, Sequence, Mapping, Callable
from omegaconf import DictConfig
import importlib
from fmod.data.batch import BatchDataset

class SRModels:

	def __init__(self, input_dataset: BatchDataset, target_dataset: BatchDataset, device: torch.device):
		self.model_config = dict( cfg().model.items() )
		self.model_name = cfg().model.name
		self.device = device
		self.datasets: Dict[str,BatchDataset] = dict( input = input_dataset, target = target_dataset )
		input_batch: Dict[str,xa.DataArray] = input_dataset.get_current_batch()
		target_batch: Dict[str, xa.DataArray] = target_dataset.get_current_batch()
		print( f" !!! Get Sample data !!! ", flush=True )
		self.sample_input:  xa.DataArray = input_batch['input']
		self.sample_target: xa.DataArray = target_batch['target']
		print(f"sample_input: shape={self.sample_input.shape}")
		print(f"sample_target: shape={self.sample_target.shape}")
		self.model_config['nchannels'] = self.sample_input.shape[1]

	def get_model(self) -> nn.Module:
		importpath = f"fmod.model.sres.{self.model_name}.network"
		model_package = importlib.import_module(importpath)
		return model_package.get_model( self.model_config ).to(self.device)





	#