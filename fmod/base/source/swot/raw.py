from fmod.base.source.loader.raw import SRRawDataLoader
import xarray as xa, math, os
from fmod.base.util.config import cfg, dateindex
from omegaconf import DictConfig, OmegaConf
from nvidia.dali import fn
from enum import Enum
from glob import glob
from typing import Any, Mapping, Sequence, Tuple, Union, List, Dict, Literal, Optional
from fmod.base.io.loader import data_suffix, path_suffix
from fmod.base.util.logging import lgm, exception_handled, log_timing
from .util import mds2d
import numpy as np

def filepath(ftype: str ) -> str:
	return f"{cfg().dataset.dataset_root}/{cfg().dataset.dataset_files[ftype]}"
class SWOTRawDataLoader(SRRawDataLoader):

	def __init__(self, task_config: DictConfig, **kwargs ):
		self.task = task_config
		self.parms = kwargs

	def load_file( self, **kwargs ) -> np.ndarray:
		for cparm, value in kwargs.items():
			cfg().dataset[cparm] = value
		var_template: np.ndarray = np.fromfile(filepath('template'), '>f4')
		var_data: np.ndarray = np.fromfile(filepath('raw'), '>f4')
		mask = var_template == 0
		var_template[~mask] = var_data
		var_template[mask] = np.nan
		sss_east, sss_west = mds2d(var_template)
		print(sss_east.shape, sss_west.shape)
		result = np.c_[sss_east, sss_west.T[::-1, :]]
		return result