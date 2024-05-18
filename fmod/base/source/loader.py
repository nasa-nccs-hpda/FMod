import xarray as xa, math, os
from datetime import datetime
from enum import Enum
from typing import Any, Mapping, Sequence, Tuple, Union, List, Dict, Literal
from fmod.base.util.ops import format_timedeltas, fmbdir
import xarray as xa
import time, numpy as np
from fmod.base.util.dates import date_list
from datetime import date
from fmod.base.util.logging import lgm, log_timing
from fmod.base.util.config import cfg
from omegaconf import DictConfig, OmegaConf
from fmod.base.util.ops import remove_filepath

class srRes(Enum):
	Low = 'lr'
	High = 'hr'

	@classmethod
	def from_config(cls, sval: str ) -> 'srRes':
		if sval == "low": return cls.Low
		if sval == "high": return cls.High
class SRDataLoader(object):

	def __init__(self, task_config: DictConfig, vres: str ):
		self.vres: srRes = srRes.from_config(vres)
		self.task = task_config

	def load_norm_data(self)-> Dict[str, xa.Dataset]:
		raise NotImplementedError("SRDataLoader:load_norm_data")

	def load_dataset(self, name: str, origin: Dict[str,int], date_range: Tuple[datetime,datetime] ):
		raise NotImplementedError("SRDataLoader:load_dataset")

	def load_const_dataset(self, origin: Tuple[int,int] ):
		raise NotImplementedError("SRDataLoader:load_const_dataset")

	@classmethod
	def rcoords( cls, dset: xa.Dataset ):
		c = dset.coords
		return '[' + ','.join([f"{k}:{c[k].size}" for k in c.keys()]) + ']'

	@classmethod
	def get_loader(cls, task_config: DictConfig, vres: str, ** kwargs ) -> 'SRDataLoader':
		if task_config.dataset == "LLC4320":
			from fmod.base.source.s3export.batch import S3ExportDataLoader
			return S3ExportDataLoader( task_config, vres, **kwargs )
		elif task_config.dataset == "merra2":
			return

class FMDataLoader(object):

	def load_norm_data(self)-> Dict[str, xa.Dataset]:
		raise NotImplementedError("SRDataLoader:load_norm_data")

	def load_dataset(self, d: date, vres: str):
		raise NotImplementedError("SRDataLoader:load_norm_data")

	def load_const_dataset(self, vres: str):
		raise NotImplementedError("SRDataLoader:load_norm_data")

	def rcoords(self, dset: xa.Dataset):
		raise NotImplementedError("SRDataLoader:rcoords")

	@classmethod
	def get_loader(cls, task_config: DictConfig, ** kwargs):
		pass