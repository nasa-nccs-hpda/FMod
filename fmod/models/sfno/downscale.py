import xarray as xa
import numpy as np
from fmod.base.util.config import cfg
from fmod.models.sfno import sht
from typing import List, Union, Tuple, Optional, Dict, Type
import torch

class SFNODownscaler(object):

	def __init__(self, target: xa.DataArray, source_shape: List[int]=None, **kwargs):
		self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		self.c = cfg().task.coords
		self.coords = target.coords
		self.dims = target.dims
		self.attrs = target.attrs
		self.tname = target.name
		self.target_shape = target.sizes[self.c['y']], target.sizes[self.c['x']]      # lat, lon6
		self.source_shape = self.target_shape if source_shape is None else source_shape    # lat, lon
		self.sht = sht.RealSHT( *self.source_shape, *self.target_shape, grid="equiangular").to(self.device)
		self.isht = sht.InverseRealSHT( *self.target_shape, *self.target_shape, grid="equiangular").to(self.device)
		self.coef: torch.Tensor = None


	def process( self, variable: xa.DataArray ) -> xa.DataArray:
		signal: torch.Tensor = torch.from_numpy(variable.values).to(self.device)
		print( f"SFNO: signal.shape={signal.shape}, source_shape={self.source_shape}, target_shape={self.target_shape}" )
		self.coef = self.sht(signal)
		print(f" ---> coef.shape={self.coef.shape}")
		downscaled: np.ndarray = self.isht( self.coef ).numpy()
		print(f" ---> downscaled.shape={downscaled.shape}")
		return xa.DataArray( data=downscaled, coords=self.coords, dims=self.dims, attrs=self.attrs, name=self.tname )