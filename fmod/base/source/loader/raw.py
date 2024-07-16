import math, os
from enum import Enum
from typing import Any, Mapping, Sequence, Tuple, Union, List, Dict, Literal
import time, numpy as np
from fmod.data.tiles import TileGrid
import xarray as xa
from fmod.base.util.logging import lgm, log_timing
from fmod.base.util.config import cfg
from omegaconf import DictConfig, OmegaConf

class SRRawDataLoader(object):

	def __init__(self, config: DictConfig, **kwargs):
		self.config = config
		self.tile_grid = TileGrid()
		self.varnames: Dict[str, str] = self.config.input_variables

	@classmethod
	def get_loader(cls, task_config: DictConfig, **kwargs) -> 'SRRawDataLoader':
		dset: str = task_config.dataset
		if dset.startswith("swot"):
			from fmod.base.source.swot.raw import SWOTRawDataLoader
			return SWOTRawDataLoader( task_config, **kwargs )

	def load_timeslice(self, **kwargs) -> xa.DataArray:
		raise NotImplementedError("SRRawDataLoader:load_timeslice")

	def get_batch_time_indices(self, **kwargs) -> xa.DataArray:
		raise NotImplementedError("SRRawDataLoader:get_batch_time_indices")

	def get_norm_stats(self) -> Dict[Tuple[str,int], Tuple[float,float]]:
		raise NotImplementedError("SRRawDataLoader:get_batch_time_indices")
