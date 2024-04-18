import numpy as np, xarray as xa
import cftime, torch
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
	def __init__( self, *args, **kwargs ):
		self.device = kwargs.pop('device', 'gpu')
		self.train_dates= kwargs.pop('train_dates', date_list(cfg_date('task'), cfg().task.max_days) )
		self.input_dataset = MERRA2Dataset(train_dates=self.train_dates, vres='low', load_inputs=True, load_base=True, load_targets=False)
		self.target_dataset = MERRA2Dataset(train_dates=self.train_dates, vres='high', load_inputs=False, load_base=False, load_targets=True)
		self.trainer = DualModelTrainer(self.input_dataset, self.target_dataset, self.device)
		_input: xa.DataArray = self.input_dataset[0][0]
		_target: xa.DataArray = self.target_dataset[0][0]
		self.input_coords = _input.coords
		self.nchannels = [ _input.shape[0], _target.shape[0]]
		lgm().log(f"Number of valid times: {self.input_dataset.length}",  )
		lgm().log(f"valid times: {self.train_dates}" )
		lgm().log(f"input_channels: {self.input_channels()}" )
		lgm().log(f"output_channels: {self.output_channels()}")

	def tensor(self, data: xa.DataArray) -> torch.Tensor:
		return torch.Tensor(data.values).to(self.device)

	def __getitem__(self, idx: int) -> Tuple[ torch.Tensor, torch.Tensor, str ]:
		[ _input, base ] = self.input_dataset[idx]
		[ target ]  = self.target_dataset[idx]
		return self.tensor(target), self.tensor(_input), ""

	def __len__(self):
		return self.input_dataset.length

	def longitude(self):
		return self.input_coords['lon']

	def latitude(self):
		return self.input_coords['lat']

	def _get_channel_meta(self, variable, level):
		return ChannelMetadata(name=variable, level=str(level))

	def input_channels(self):
		"""Metadata for the input channels. A list of dictionaries, one for each channel"""
		return self.nchannels[0]

	def output_channels(self):
		"""Metadata for the output channels. A list of dictionaries, one for each channel"""
		return self.nchannels[1]


	def time(self):
		return self.input_coords['time']

	def image_shape(self):
		return [ self.input_coords['lat'].size, self.input_coords['lon'].size ]

	# def _select_norm_channels(self, means, stds, channels):
	# 	if channels is not None:
	# 		means = means[channels]
	# 		stds = stds[channels]
	# 	return (means, stds)

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

def get_dataset( *args, **kwargs ):
    return M2DownscalingDataset( *args, **kwargs )

