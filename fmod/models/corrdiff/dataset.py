import numpy as np, xarray as xa
import cftime
from fmod.models.corrdiff.nvidia.datasets.base import DownscalingDataset, ChannelMetadata
from typing import Any, Dict, List, Tuple, Type, Optional, Union, TypeVar
from fmod.pipeline.merra2 import MERRA2Dataset
from fmod.base.util.logging import lgm
T_co = TypeVar('T_co', covariant=True)
from datetime import date
from fmod.base.util.dates import date_list
from fmod.base.util.config import configure, cfg, cfg_date, cfg2args, pp
from fmod.pipeline.trainer import DualModelTrainer
from fmod.pipeline.merra2 import MERRA2Dataset


class M2DownscalingDataset(DownscalingDataset):
	path: str
	def __init__( self, path="", **kwargs ):
		self.path=path
		device = kwargs.pop('device', 'gpu')
		self.train_dates= kwargs.pop('train_dates', date_list(cfg_date('task'), cfg().task.max_days) )
		self.input_dataset = MERRA2Dataset(train_dates=self.train_dates, vres='low', load_inputs=True, load_base=True, load_targets=False)
		self.target_dataset = MERRA2Dataset(train_dates=self.train_dates, vres='high', load_inputs=False, load_base=False, load_targets=True)
		self.trainer = DualModelTrainer(self.input_dataset, self.target_dataset, device)
		lgm().log(f"Number of valid times: {self.input_dataset.length}",  )
		lgm().log(f"valid times: {self.train_dates}" )
		lgm().log(f"input_channels: {self.input_channels()}" )
		lgm().log(f"output_channels: {self.output_channels()}")

	def __getitem__(self, idx: int):
		ds_input: List[xa.Dataset] = self.input_dataset[idx]
		ds_target: List[xa.Dataset]  = self.input_dataset[idx]
		return ds_target, ds_input, ""


		# target = self.normalize_output(target[None, ...])[0]
		# input = self.normalize_input(input[None, ...])[0]
		#
		# return target, input, label

	"""
	Takes in np array of size (n_history+1, c, h, w) and returns torch tensor of
	size ((n_channels*(n_history+1), crop_size_x, crop_size_y)
	"""

	def __len__(self):
		return self.input_dataset.length

	def longitude(self):
		return None

	def latitude(self):
		return None

	def _get_channel_meta(self, variable, level):
		return ChannelMetadata(name=variable, level=str(level))

	def input_channels(self):
		"""Metadata for the input channels. A list of dictionaries, one for each channel"""
		return 0

	def output_channels(self):
		"""Metadata for the output channels. A list of dictionaries, one for each channel"""
		return 0

	def _read_time(self):
		"""The vector of time coordinate has length (self)"""
		return None

	def time(self):
		"""The vector of time coordinate has length (self)"""
		return None

	def image_shape(self):
		"""Get the shape of the image (same for input and output)."""
		return None

	def _select_norm_channels(self, means, stds, channels):
		if channels is not None:
			means = means[channels]
			stds = stds[channels]
		return (means, stds)

	# def normalize_input(self, x, channels=None):
	# 	"""Convert input from physical units to normalized data."""
	# 	norm = self._select_norm_channels(
	# 		self.group["era5_center"], self.group["era5_scale"], channels
	# 	)
	# 	return normalize(x, *norm)
	#
	# def denormalize_input(self, x, channels=None):
	# 	"""Convert input from normalized data to physical units."""
	# 	norm = self._select_norm_channels(
	# 		self.group["era5_center"], self.group["era5_scale"], channels
	# 	)
	# 	return denormalize(x, *norm)
	#
	# def normalize_output(self, x, channels=None):
	# 	"""Convert output from physical units to normalized data."""
	# 	norm = self.get_target_normalization(self.group)
	# 	norm = self._select_norm_channels(*norm, channels)
	# 	return normalize(x, *norm)
	#
	# def denormalize_output(self, x, channels=None):
	# 	"""Convert output from normalized data to physical units."""
	# 	norm = self.get_target_normalization(self.group)
	# 	norm = self._select_norm_channels(*norm, channels)
	# 	return denormalize(x, *norm)

	def info(self):
		return { "target_normalization": None, "input_normalization": None }

