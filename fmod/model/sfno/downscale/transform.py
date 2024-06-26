import xarray as xa
import numpy as np
from fmod.base.util.config import cfg
from fmod.model.sfno import sht
from typing import List, Union, Tuple, Optional, Dict, Type
import torch

class SHTransform(object):

	def __init__(self, target: xa.DataArray, source: xa.DataArray=None, **kwargs):
		self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		self.c = cfg().task.coords
		self.coords = target.coords
		self.dims = target.dims
		self.attrs = target.attrs
		self.tname = target.name
		self.grid = kwargs.get( 'grid', "equiangular" )
		self.method = kwargs.get('method', "rescale")
		self.target_shape: Tuple[int,int] = self.gshape(target)
		self.source_shape: Tuple[int,int] = self.target_shape if source is None else self.gshape(source)
		ntheta_s, nlambda_s = self.source_shape[0], self.source_shape[1]
		ntheta_t, nlambda_t = self.target_shape[0], self.target_shape[1]
		print(f" SHTransform: source_shape = {self.source_shape}, target_shape = {self.target_shape} ")
		# self.sht = sht.RealSHT( nlat=self.source_shape[0], nlon=self.source_shape[1], lmax=self.target_shape[0], mmax=self.target_shape[1], grid=self.grid ).to(self.device)
		# self.isht = sht.InverseRealSHT( *self.target_shape, *self.target_shape, grid=self.grid ).to(self.device)
		self.sht = sht.RealSHT( ntheta_s, nlambda_s, grid=self.grid ).to(self.device)
		self.isht = sht.InverseRealSHT( ntheta_t, nlambda_t, ntheta_s, nlambda_s, grid=self.grid ).to(self.device)
		self.coef: torch.Tensor = None

	def gshape(self, grid: xa.DataArray ) -> Tuple[int,int]:     # lat, lon
		return grid.sizes[self.c['y']], grid.sizes[self.c['x']]

	def process( self, variable: xa.DataArray ) -> xa.DataArray:
		signal: torch.Tensor = torch.from_numpy(variable.values).to(self.device)
		print( f"SFNO: signal.shape={signal.shape}, source_shape={self.source_shape}, target_shape={self.target_shape}" )
		self.coef = self.sht(signal)
		print(f" ---> coef.shape={self.coef.shape}")
		downscaled: np.ndarray = self.isht( self.coef ).cpu().numpy()
		print(f" ---> downscaled.shape={downscaled.shape}")
		return xa.DataArray( data=downscaled, coords=self.coords, dims=self.dims, attrs=self.attrs, name=self.tname )