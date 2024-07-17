from fmod.base.source.loader.raw import SRRawDataLoader
import xarray as xa, math, os, pickle
from fmod.base.util.config import cfg, dateindex
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

class NormData:

	def __init__(self, itile: int):
		self.itile = itile
		self.means: List[float] = []
		self.stds: List[float] = []

	def add_entry(self, tiles_data: xa.DataArray ):
		tdata: np.ndarray = tiles_data.isel(tiles=self.itile).values.squeeze()
		self.means.append(tdata.mean())
		self.stds.append(tdata.std())

	def get_norm_stats(self) -> Tuple[float,float]:
		return  np.array(self.means).mean(), np.array(self.stds).mean()

class SWOTRawDataLoader(SRRawDataLoader):

	def __init__(self, task_config: DictConfig, **kwargs ):
		super(SWOTRawDataLoader, self).__init__(task_config)
		self.parms = kwargs
		self.dataset = DictConfig.copy( cfg().dataset )
		self.tset: TSet = None
		self.time_index: int = -1
		self.timeslice: xa.DataArray = None
		self.norm_data_file = f"{cfg().platform.cache}/norm_data/norms/norms.{cfg().task.training_version}.pkl"
		self._norm_stats: Dict[Tuple[str,int], Tuple[float,float]]  = None
		os.makedirs( os.path.dirname(self.norm_data_file), 0o777, exist_ok=True )

	def _write_norm_stats(self, norm_stats: Dict[Tuple[str,int],Tuple[float,float]] ):
		print("_write_norm_stats")
		with open(self.norm_data_file, 'wb') as file_handle:
			pickle.dump(list(norm_stats.items()), file_handle, protocol=pickle.HIGHEST_PROTOCOL)

	def _read_norm_stats(self) -> Optional[Dict[Tuple[str,int],Tuple[float,float]]]:
		print("_read_norm_stats")
		if os.path.isfile(self.norm_data_file):
			with open(self.norm_data_file, 'rb') as file_handle:
				norm_data = pickle.load(file_handle)
				return dict(norm_data)

	def _compute_normalization(self) -> Dict[Tuple[str,int], Tuple[float,float]]:
		print("_compute_normalization")
		time_indices = self.get_batch_time_indices()
		norm_data: Dict[Tuple[str,int], NormData] = {}
		for varname in self.varnames:
			for tidx in time_indices:
				file_data: np.ndarray = self.load_file( varname, tidx )
				tiles_data: xa.DataArray = self.get_tiles(file_data)
				for itile in range(tiles_data.sizes['tiles']):
					norm_entry: NormData = norm_data.setdefault((varname,itile), NormData(itile))
					norm_entry.add_entry( tiles_data )
		vtstats: Dict[str,Dict[int,Tuple[float,float]]] = {}
		for (varname,itile), nd in norm_data.items():
			tmean, tstd = nd.get_norm_stats()
			ns = vtstats.setdefault(varname,{})
			ns[itile] = (tmean, tstd)
		for varname in self.varnames:
			vs = vtstats[varname]
			idxs = list(vs.keys())
			nstats = np.array( list(vs.values()) )
			print( f"idxs={idxs}" )
			print( f"nstats.shape={nstats.shape}" )
		return vtstats

	def _get_norm_stats(self) -> Dict[Tuple[str,int], Tuple[float,float]]:
		print("_get_norm_stats")
		norm_stats: Dict[Tuple[str,int], Tuple[float,float]] = self._read_norm_stats()
		if norm_stats is None:
			norm_stats = self._compute_normalization()
			self._write_norm_stats(norm_stats)
		return norm_stats

	def norm_stats(self) -> Dict[Tuple[str,int], Tuple[float,float]]:
		print(f"norm_stats: compute={self._norm_stats is None}")
		if self._norm_stats is None:
			self._norm_stats = self._get_norm_stats()
		return self._norm_stats

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
			lgm().log( f"\nLoaded timeslice{self.timeslice.dims} shape={self.timeslice.shape}, mean={self.timeslice.values.mean():.2f}, std={self.timeslice.values.std():.2f}", display=True)
			self.time_index = time_index
		return self.select_batch( tile_range )

	def select_batch( self, tile_range: Tuple[int,int]  ) -> Optional[xa.DataArray]:
		ntiles: int = self.timeslice.shape[0]
		if tile_range[0] < ntiles:
			slice_end = min(tile_range[1], ntiles)
			batch: xa.DataArray =  self.timeslice.isel( tiles=slice(tile_range[0],slice_end) )
			bmean, bstd = batch.mean( dim=["x", "y"], skipna=True, keep_attrs=True ), batch.std( dim=["x", "y"], skipna=True, keep_attrs=True )
			batch = (batch-bmean)/bstd
			lgm().log(f"\n select_batch[{self.time_index}]{batch.dims}{batch.shape}: tile_range= {(tile_range[0], slice_end)}", display=True )
			print(f" **mean: {batch.mean(dim=['x','y']).squeeze().values.tolist()}" )
			print(f" **std:  {batch.std( dim=['x','y']).squeeze().values.tolist()}")
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