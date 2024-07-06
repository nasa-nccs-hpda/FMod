from fmod.base.source.loader.raw import SRRawDataLoader
import xarray as xa, math, os
from fmod.base.util.config import cfg, dateindex
from fmod.data.batch import TileGrid
from fmod.base.io.loader import ncFormat, TSet
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

	def __init__(self, **kwargs ):
		self.parms = kwargs
		self.tile_grid = TileGrid()


	def load_file( self, **kwargs ) -> xa.DataArray:
		for cparm, value in kwargs.items():
			cfg().dataset[cparm] = value
		var_template: np.ndarray = np.fromfile(filepath('template'), '>f4')
		var_data: np.ndarray = np.fromfile(filepath('raw'), '>f4')
		mask = (var_template == 0)
		var_template[~mask] = var_data
		var_template[mask] = np.nan
		sss_east, sss_west = mds2d(var_template)
		print(sss_east.shape, sss_west.shape)
		result = np.expand_dims( np.c_[sss_east, sss_west.T[::-1, :]], 0)
		return xa.DataArray( result, dims=["channel","y","x"], name=kwargs.get('varname','') )

	def get_tiles(self, raw_data: np.ndarray, **kwargs):
		ts = self.tile_grid.get_full_tile_size()
		tile_grid: Dict[str, int] = self.tile_grid.get_grid_shape( dict(x=raw_data.shape[-1], y=raw_data.shape[-2]) )
		sss_reshape = np.swapaxes(raw_data.reshape(ny_n, ngrid, nx_n, ngrid), 1, 2).reshape(ny_n * nx_n, ngrid, ngrid)  # reshape the global data to ny_n*nx_n stamps each of which has ngrid x ngrid shape
		# Find the regions that hava no land (nan) values
		msk = np.isfinite(sss_reshape.mean(axis=-1).mean(axis=-1))
		sss_stamps = sss_reshape[msk, ...]