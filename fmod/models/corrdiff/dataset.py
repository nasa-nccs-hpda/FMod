import numpy as np, xarray as xa
import cftime
from fmod.models.corrdiff.nvidia.datasets.base import DownscalingDataset, ChannelMetadata
from typing import Any, Dict, List, Tuple, Type, Optional, Union, TypeVar
from fmod.pipeline.merra2 import MERRA2Dataset
from fmod.base.util.logging import lgm
T_co = TypeVar('T_co', covariant=True)
from datetime import date
from fmod.base.util.dates import year_range

class M2DownscalingDataset(DownscalingDataset):
	path: str

	def __init__( self, **kwargs ):
		self.ds_lowres = MERRA2Dataset( vres="low", **kwargs )
		self.ds_highres = MERRA2Dataset (vres="high", **kwargs )
		lgm().log("Number of valid times: %d", self.ds_lowres.length )
		lgm().log("input_channels:%s", self.input_channels())
		lgm().log("output_channels:%s", self.output_channels())


	def __getitem__(self, idx: int):
		ds_target: xa.Dataset = self.ds_highres.__getitem__(idx)
		ds_input: xa.Dataset  = self.ds_lowres.__getitem__(idx)

		return ds_target.values(), ds_input.values(), ""


		# target = self.normalize_output(target[None, ...])[0]
		# input = self.normalize_input(input[None, ...])[0]
		#
		# return target, input, label

	"""
	Takes in np array of size (n_history+1, c, h, w) and returns torch tensor of
	size ((n_channels*(n_history+1), crop_size_x, crop_size_y)
	"""

	def __len__(self):
		return self.ds_lowres.length

	def longitude(self):
		return None

	def latitude(self):
		return None

	def _get_channel_meta(self, variable, level):
		return ChannelMetadata(name=variable, level=str(level))

	def input_channels(self):
		"""Metadata for the input channels. A list of dictionaries, one for each channel"""
		return None

	def output_channels(self):
		"""Metadata for the output channels. A list of dictionaries, one for each channel"""
		return None

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

