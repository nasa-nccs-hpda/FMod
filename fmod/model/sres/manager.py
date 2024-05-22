import logging, torch, math
from fmod.base.util.logging import lgm, exception_handled, log_timing
import torch.nn as nn
import xarray as xa
import os, time, numpy as np
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
		self.target_variables = cfg().task.target_variables
		self.datasets: Dict[str,BatchDataset] = dict( input = input_dataset, target = target_dataset )
		self.sample_input:  xa.DataArray = input_dataset.get_current_batch_array()
		self.sample_target: xa.DataArray = target_dataset.get_current_batch_array()
		self.cids: List[int] = self.get_channel_idxs(self.target_variables)
		print(f"sample_input: shape={self.sample_input.shape}")
		print(f"sample_target: shape={self.sample_target.shape}")
		self.model_config['nchannels'] = self.sample_input.sizes['channels']

	def get_channel_idxs(self, channels: List[str], dstype: str = "input") -> List[int]:
		return self.datasets[dstype].get_channel_idxs(channels)

	def get_sample_target(self) -> xa.DataArray:
		result =  self.sample_target.isel(channels=self.cids) if (len(self.cids) < self.sample_target.sizes['channels']) else self.sample_target
		print(f" !!! Get Sample target !!! cids={self.cids}: sample_target{self.sample_target.dims}{self.sample_target.shape}, result{result.shape}", flush=True)
		return result
	def get_sample_input(self) -> xa.DataArray:
		result =  self.sample_input.isel(channels=self.cids) if (len(self.cids) < self.sample_input.sizes['channels']) else self.sample_input
		print(f" !!! Get Sample input !!! cids={self.cids}: sample_input{self.sample_input.dims}{self.sample_input.shape}, result{result.shape}", flush=True)
		return result

	def filter_targets(self, data_array: np.ndarray ) -> np.ndarray:
		return np.take( data_array, self.cids, axis=1 )

	def get_model(self) -> nn.Module:
		importpath = f"fmod.model.sres.{self.model_name}.network"
		model_package = importlib.import_module(importpath)
		return model_package.get_model( self.model_config ).to(self.device)





	#