import logging, torch, math
from fmod.base.util.logging import lgm, exception_handled, log_timing
import torch.nn as nn
import xarray as xa
import os, time
from typing import Any, Dict, List, Tuple, Type, Optional, Union, Sequence, Mapping, Callable
from omegaconf import DictConfig
from fmod.base.io.loader import BaseDataset

class SRModels:

	def __init__(self, input_dataset: BaseDataset, model_config: DictConfig, device: torch.device):
		self.model_config = dict(*model_config)
		self.device = device
		train_data: Dict[str, xa.DataArray] = input_dataset.get_current_batch()
		self.sample_input:  xa.DataArray = train_data['input']
		self.sample_target: xa.DataArray = train_data['target']
		print(f"sample_input: shape={self.sample_input.shape}")
		print(f"sample_target: shape={self.sample_target.shape}")
		self.model_config['nchannels'] = self.sample_input.shape[1]

	def get_model(self):
		modelId = self.model_config['name']
		if modelId == "mscnn":
			from fmod.models.sres.mscnn.network import get_model, MSCNN
			return get_model( self.model_config ).to(self.device)




	#