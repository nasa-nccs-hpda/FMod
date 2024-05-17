import xarray as xa, math, os
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

class SRDataLoader(object):

	def load_norm_data(self):
		raise NotImplementedError("SRDataLoader:load_norm_data")

	def load_dataset(self, d: date, vres: str):
		raise NotImplementedError("SRDataLoader:load_norm_data")

	def load_const_dataset(self, vres: str):
		raise NotImplementedError("SRDataLoader:load_norm_data")

	def rcoords(self, dset: xa.Dataset):
		raise NotImplementedError("SRDataLoader:rcoords")

	@classmethod
	def get_loader(cls, task_config: DictConfig, ** kwargs):
		if task_config.dataset == "LLC4320":
			from fmod.base.source.s3export.batch import S3ExportDataLoader
			return S3ExportDataLoader( task_config, ** kwargs )
		elif task_config.dataset == "LLC4320":
			return

class FMDataLoader(object):

	def load_norm_data(self):
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