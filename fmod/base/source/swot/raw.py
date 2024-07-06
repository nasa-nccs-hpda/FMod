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

def filepath(ftype: str) -> str:
	return f"{cfg().dataset.dataset_root}/{cfg().dataset.dataset_files[ftype]}"

class SWOTRawDataLoader(SRRawDataLoader):

	def __init__(self, task_config: DictConfig, **kwargs ):
		super(SWOTRawDataLoader, self).__init__(task_config)
		self.parms = kwargs
		self.dataset = DictConfig.copy( cfg().dataset )

	def load_file( self, **kwargs ) -> np.ndarray:
		for cparm, value in kwargs.items():
			cfg().dataset[cparm] = value
		var_template: np.ndarray = np.fromfile( filepath('template'), '>f4' )
		var_data: np.ndarray = np.fromfile( filepath('raw'), '>f4' )
		mask = (var_template == 0)
		var_template[~mask] = var_data
		var_template[mask] = np.nan
		sss_east, sss_west = mds2d(var_template)
		print(sss_east.shape, sss_west.shape)
		result = np.expand_dims( np.c_[sss_east, sss_west.T[::-1, :]], 0)
		print( f"load_file result = {result.shape}")
		return result

	def load_timeslice( self, **kwargs ) -> xa.DataArray:
		vardata: List[np.ndarray] = [ self.load_file( varname=varname, **kwargs ) for varname in self.varnames ]
		return self.get_tiles( np.concatenate(vardata,axis=0) )

		#return xa.DataArray( result, dims=["channel","y","x"], name=kwargs.get('varname','') )

	def get_tiles(self, raw_data: np.ndarray) -> xa.DataArray:       # dims=["channel","y","x"]
		tsize: Dict[str, int] = self.tile_grid.get_full_tile_size()
		print(f"get_tiles(raw_data) = {raw_data.shape}, ndim = {raw_data.ndim}")
		if raw_data.ndim == 2: raw_data = np.expand_dims( raw_data, 0 )
		ishape = dict(c=raw_data.shape[0], y=raw_data.shape[1], x=raw_data.shape[2])
		print(f"raw_data = {raw_data.shape}, ishape = {ishape}")
		grid_shape: Dict[str, int] = self.tile_grid.get_grid_shape( ishape )
		roi: Dict[str, Tuple[int,int]] = self.tile_grid.get_active_region(ishape)
		print( f"roi = {roi}")
		region_data: np.ndarray = raw_data[..., roi['y'][0]:roi['y'][1], roi['x'][0]:roi['x'][1]]
		print(f"region_data = {region_data.shape}")
		tile_data = region_data.reshape( ishape['c'], grid_shape['y'], tsize['y'], grid_shape['x'], tsize['x'] )
		print(f"tile_data = {tile_data.shape}")
		tiles = np.swapaxes(tile_data, 2, 3).reshape( ishape['c'] * grid_shape['y'] * grid_shape['x'], tsize['y'], tsize['x'])
		print(f"tiles = {tiles.shape}")
		msk = np.isfinite(tiles.mean(axis=-1).mean(axis=-1))
		ntiles = np.count_nonzero(msk)
		print( f"msk = {msk.shape}, ntiles = {ntiles}"  )
		print( tiles.shape )
		result = np.compress( msk, tiles, 0)
		result = result.reshape( ntiles//ishape['c'], ishape['c'], tsize['y'], tsize['x'] )
		return xa.DataArray(result, dims=["sample","channel", "y", "x"])