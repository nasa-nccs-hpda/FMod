import xarray as xa
import numpy as np
from fmod.base.util.config import cfg
from torch_harmonics import *
import torch

class SFNODownscaler(object):

	def __init__(self, target: xa.DataArray, **kwargs):
		self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		self.c = cfg().task.coords
		self.coords = target.coords
		self.dims = target.dims
		self.attrs = target.attrs
		self.tname = target.name
		nlat, nlon = target.sizes[self.c['y']], target.sizes[self.c['x']]
		n_theta, n_lambda = nlat, nlon
		self.sht = RealSHT(n_theta, n_lambda, grid="equiangular").to(self.device)
		self.isht = InverseRealSHT(nlat, nlon, grid="equiangular").to(self.device)
		self.coef: torch.Tensor = None


	def process( self, variable: xa.DataArray ) -> xa.DataArray:
		signal: torch.Tensor = torch.from_numpy(variable.values).to(self.device)
		self.coef = self.sht(signal)
		downscaled: np.ndarray = self.isht( self.coef ).numpy()
		return xa.DataArray( data=downscaled, coords=self.coords, dims=self.dims, attrs=self.attrs, name=self.tname )