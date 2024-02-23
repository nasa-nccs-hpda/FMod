import xarray as xa
import numpy as np
from fmod.base.util.config import cfg
from fmod.models.sfno import sht
import torch

class SFNODownscaler(object):

	def __init__(self, target: xa.DataArray, **kwargs):
		self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		self.c = cfg().task.coords
		self.coords = target.coords
		self.dims = target.dims
		self.attrs = target.attrs
		self.tname = target.name
		self.nlat, self.nlon = target.sizes[self.c['y']], target.sizes[self.c['x']]
		self.n_theta, self.n_lambda = self.nlat, self.nlon
		self.sht = sht.RealSHT(self.n_theta, self.n_lambda, grid="equiangular").to(self.device)
		self.isht = sht.InverseRealSHT(self.nlat, self.nlon, grid="equiangular").to(self.device)
		self.coef: torch.Tensor = None


	def process( self, variable: xa.DataArray ) -> xa.DataArray:
		signal: torch.Tensor = torch.from_numpy(variable.values).to(self.device)
		print( f"SFNO: signal.shape={signal.shape}, nlat,nlon={(self.nlat, self.nlon)}, n_theta,n_lambda={(self.n_theta,self.n_lambda)}" )
		self.coef = self.sht(signal)
		print(f" ---> coef.shape={self.coef.shape}")
		downscaled: np.ndarray = self.isht( self.coef ).numpy()
		print(f" ---> downscaled.shape={downscaled.shape}")
		return xa.DataArray( data=downscaled, coords=self.coords, dims=self.dims, attrs=self.attrs, name=self.tname )