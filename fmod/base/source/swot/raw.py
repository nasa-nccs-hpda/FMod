from fmod.base.source.loader.raw import SRRawDataLoader
import xarray as xa, math, os
from fmod.base.util.config import cfg, dateindex
from fmod.data.tiles import TileGrid
from fmod.base.io.loader import ncFormat, TSet
from omegaconf import DictConfig, OmegaConf
from nvidia.dali import fn
from enum import Enum
from glob import glob
from typing import Any, Mapping, Sequence, Tuple, Union, List, Dict, Literal, Optional
from fmod.base.io.loader import data_suffix, path_suffix
from fmod.base.util.logging import lgm, exception_handled, log_timing
from .util import mds2d
from glob import glob
from parse import parse
import numpy as np

def filepath() -> str:
	return f"{cfg().dataset.dataset_root}/{cfg().dataset.dataset_files}"

def template() -> str:
	return f"{cfg().dataset.dataset_root}/{cfg().dataset.template}"

class SWOTRawDataLoader(SRRawDataLoader):

	def __init__(self, task_config: DictConfig, **kwargs ):
		super(SWOTRawDataLoader, self).__init__(task_config)
		self.parms = kwargs
		self.dataset = DictConfig.copy( cfg().dataset )
		self.tset: TSet = None
		self.time_index: int = -1
		self.timeslice: xa.DataArray = None

	def get_batch_time_indices(self):
		cfg().dataset.index = "*"
		cfg().dataset['varname'] = list(self.varnames.keys())[0]
		files = [ fpath.split("/")[-1] for fpath in  glob( filepath() ) ]
		template = filepath().replace("*",'{}').split("/")[-1]
		indices = [ int(parse(template,f)[0]) for f in files ]
		return indices

	def load_file( self,  varname: str, time_index: int ) -> np.ndarray:
		for cparm, value in dict(varname=varname, index=time_index).items():
			cfg().dataset[cparm] = value
		var_template: np.ndarray = np.fromfile(template(), '>f4')
		var_data: np.ndarray = np.fromfile(filepath(), '>f4')
		mask = (var_template != 0)
		print( f" *** load_file: var_template{var_template.shape} var_data{var_data.shape} mask nz={np.count_nonzero(mask)}, file={filepath()}")
		print(f"   >>> data file={filepath()}")
		print(f"   >>> template file={template()}")
		var_template[mask] = var_data
		var_template[~mask] = np.nan
		sss_east, sss_west = mds2d(var_template)
		result = np.expand_dims( np.c_[sss_east, sss_west.T[::-1, :]], 0)
		lgm().log( f"load_file result = {result.shape}")
		return result

	def load_batch( self, tile_range: Tuple[int,int], time_index: int  ) -> Optional[xa.DataArray]:
		if time_index != self.time_index:
			vardata: List[np.ndarray] = [ self.load_file( varname, time_index ) for varname in self.varnames ]
			self.timeslice = self.get_tiles( np.concatenate(vardata,axis=0) )
			print( f"\nLoaded timeslice{self.timeslice.dims} shape={self.timeslice.shape}, mean={self.timeslice.values.mean():.2f}, std={self.timeslice.values.std():.2f}")
			self.time_index = time_index
		return self.select_batch( tile_range )

	def select_batch( self, tile_range: Tuple[int,int]  ) -> Optional[xa.DataArray]:
		ntiles: int = self.timeslice.shape[0]
		if tile_range[0] < ntiles:
			slice_end = min(tile_range[1], ntiles)
			batch: xa.DataArray =  self.timeslice.isel( tiles=slice(tile_range[0],slice_end) )
			bmean, bstd = batch.mean( dim=["x", "y"], skipna=True, keep_attrs=True ), batch.std( dim=["x", "y"], skipna=True, keep_attrs=True )
			batch = (batch-bmean)/bstd
			lgm().log(f"\nselect_batch[{self.time_index}]{batch.dims}{batch.shape}: tile_range= {(tile_range[0], slice_end)}, mean={batch.values.mean():.2f}, std={batch.values.std():.2f}",display=True)
			return batch

	def get_tiles(self, raw_data: np.ndarray) -> xa.DataArray:
		tsize: Dict[str, int] = self.tile_grid.get_full_tile_size()
		if raw_data.ndim == 2: raw_data = np.expand_dims( raw_data, 0 )
		ishape = dict(c=raw_data.shape[0], y=raw_data.shape[1], x=raw_data.shape[2])
		grid_shape: Dict[str, int] = self.tile_grid.get_grid_shape( image_shape=ishape )
		roi: Dict[str, Tuple[int,int]] = self.tile_grid.get_active_region(image_shape=ishape)
		region_data: np.ndarray = raw_data[..., roi['y'][0]:roi['y'][1], roi['x'][0]:roi['x'][1]]
		tile_data = region_data.reshape( ishape['c'], grid_shape['y'], tsize['y'], grid_shape['x'], tsize['x'] )
		tiles = np.swapaxes(tile_data, 2, 3).reshape( ishape['c'] * grid_shape['y'] * grid_shape['x'], tsize['y'], tsize['x'])
		msk = np.isfinite(tiles.mean(axis=-1).mean(axis=-1))
		tile_idxs = np.arange(tiles.shape[0])[msk]
		ntiles = np.count_nonzero(msk)
		result = np.compress( msk, tiles, 0)
		result = result.reshape( ntiles//ishape['c'], ishape['c'], tsize['y'], tsize['x'] )
		print( f"get_tiles: shape = {result.shape}")
		return xa.DataArray(result, dims=["tiles", "channels", "y", "x"], coords=dict(tiles=tile_idxs, channels=np.array(self.varnames) ) )