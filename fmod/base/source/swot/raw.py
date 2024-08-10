from fmod.base.source.loader.raw import SRRawDataLoader
import xarray as xa, math, os, pickle
from fmod.base.util.config import cfg, config
from fmod.base.io.loader import ncFormat, TSet
from omegaconf import DictConfig, OmegaConf
from xarray.core.dataset import DataVariables
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

STATS = ['mean', 'var', 'max', 'min']
def xanorm( ndata: Dict[int, np.ndarray] ) -> xa.DataArray:
	npdata = np.stack( list(ndata.values()), axis=0 )
	return xa.DataArray( npdata, dims=['tiles','stat'], coords=dict(tiles=list(ndata.keys()), stat=STATS))

def globalize_norm( data: xa.DataArray ) -> xa.DataArray:
	results = []
	for stat in STATS:
		dslice = data.sel(stat=stat)
		if   stat == 'max': results.append(dslice.max())
		elif stat == 'min': results.append(dslice.min())
		else:               results.append(dslice.mean())
	return xa.DataArray( results, dims=['stat'], coords=dict(stat=STATS))

def filepath() -> str:
	return f"{cfg().dataset.dataset_root}/{cfg().dataset.dataset_files}"

def template() -> str:
	return f"{cfg().dataset.dataset_root}/{cfg().dataset.template}"

def subset_roi( global_data: np.ndarray ) -> np.ndarray:
	roi = cfg().dataset.get('roi',None)
	if roi is None: return global_data
	x0, xs = roi.get('x0',0), roi.get( 'xs', global_data.shape[-1] )
	y0, ys = roi.get('y0', 0), roi.get('ys', global_data.shape[-2])
	return global_data[..., y0:y0+ys, x0:x0+xs]
class NormData:

	def __init__(self, itile: int):
		self.itile = itile
		self.means: List[float] = []
		self.vars: List[float] = []
		self.max: float = -float("inf")
		self.min: float = float("inf")

	def add_entry(self, tiles_data: xa.DataArray ):
		tdata: np.ndarray = tiles_data.isel(tiles=self.itile).values.squeeze()
		self.means.append(tdata.mean())
		self.vars.append(tdata.var())
		self.max = max( self.max, tdata.max() )
		self.min = min( self.min, tdata.min() )

	def get_norm_stats(self) -> np.ndarray:
		return  np.array( [ np.array(self.means).mean(), np.array(self.vars).mean(), np.array(self.max).max(), np.array(self.min).min() ] )

class SWOTRawDataLoader(SRRawDataLoader):

	def __init__(self, task_config: DictConfig, **kwargs ):
		super(SWOTRawDataLoader, self).__init__(task_config)
		self.parms = kwargs
		self.dataset = DictConfig.copy( cfg().dataset )
		self.tset: Optional[TSet] = None
		self.time_index: int = -1
		self.timeslice: Optional[xa.DataArray] = None
		self.norm_data_file = f"{cfg().platform.cache}/norm_data/norms/norms.{config()['dataset']}.nc"
		self._norm_stats: Optional[xa.Dataset]  = None
		os.makedirs( os.path.dirname(self.norm_data_file), 0o777, exist_ok=True )

	def _write_norm_stats(self, norm_stats: xa.Dataset ):
		print(f"Writing norm data to {self.norm_data_file}")
		norm_stats.to_netcdf( self.norm_data_file, format="NETCDF4", mode="w" )

	def _read_norm_stats(self) -> Optional[xa.Dataset]:
		if os.path.exists(self.norm_data_file):
			print( f"Reading norm data from {self.norm_data_file}")
			norm_stats: xa.Dataset = xa.open_dataset(self.norm_data_file, engine='netcdf4')
			if 'tile' in norm_stats.coords: norm_stats = norm_stats.rename( dict(tile="tiles") )
			return norm_stats

	def _compute_normalization(self) -> xa.Dataset:
		time_indices = self.get_batch_time_indices()
		norm_data: Dict[Tuple[str,int], NormData] = {}
		print( f"Computing norm stats (no stats file found at {self.norm_data_file})")
		for tidx in time_indices:
			vardata: List[np.ndarray] = [ self.load_file( varname, tidx ) for varname in self.varnames ]
			tiles_data: xa.DataArray = self.get_tiles( vardata )
			for itile in range(tiles_data.sizes['tiles']):
				for varname in self.varnames:
					var_ndata = tiles_data.sel(channels=varname)
					norm_entry: NormData = norm_data.setdefault((varname,itile), NormData(itile))
					norm_entry.add_entry( var_ndata )
		vtstats: Dict[str,Dict[int,np.ndarray]] = {}
		for (varname,itile), nd in norm_data.items():
			nstats: np.ndarray = nd.get_norm_stats()
			ns = vtstats.setdefault(varname,{})
			ns[itile] = nstats
		return xa.Dataset( { vn: xanorm(ndata) for vn, ndata in vtstats.items() } )

	def _get_norm_stats(self) -> xa.Dataset:
		norm_stats: xa.Dataset = self._read_norm_stats()
		if norm_stats is None:
			norm_stats: xa.Dataset = self._compute_normalization()
			self._write_norm_stats(norm_stats)
		return norm_stats

	@property
	def norm_stats(self) -> xa.Dataset:
		if self._norm_stats is None:
			self._norm_stats = self._get_norm_stats()
		return self._norm_stats

	@property
	def global_norm_stats(self) -> xa.Dataset:
		return self.norm_stats.map( globalize_norm )

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
		var_template[mask] = var_data
		var_template[~mask] = np.nan
		sss_east, sss_west = mds2d(var_template)
		result = np.expand_dims( np.c_[sss_east, sss_west.T[::-1, :]], 0)
		roi_data = subset_roi(result)
		lgm().log( f" *** load_file: var_template{var_template.shape} var_data{var_data.shape} mask nz={np.count_nonzero(mask)}, result{roi_data.shape}, file={filepath()}", display=True)
		return roi_data

	def load_timeslice(self, time_index: int, **kwargs) -> xa.DataArray:
		if time_index != self.time_index:
			vardata: List[np.ndarray] = [ self.load_file( varname, time_index ) for varname in self.varnames ]
			self.timeslice = self.get_tiles( vardata )
			lgm().log( f"\nLoaded timeslice{self.timeslice.dims} shape={self.timeslice.shape}, mean={np.nanmean(self.timeslice.values):.2f}, std={np.nanstd(self.timeslice.values):.2f}", display=True)
			self.time_index = time_index
		return self.timeslice

	def load_batch( self, tile_range: Tuple[int,int] ) -> Optional[xa.DataArray]:
		return self.select_batch( tile_range )

	def select_batch( self, tile_range: Tuple[int,int]  ) -> Optional[xa.DataArray]:
		ntiles: int = self.timeslice.shape[0]
		if tile_range[0] < ntiles:
			slice_end = min(tile_range[1], ntiles)
			batch: xa.DataArray =  self.timeslice.isel( tiles=slice(tile_range[0],slice_end) )
			lgm().log(f" select_batch[{self.time_index}]{batch.dims}{batch.shape} from timeslice{self.timeslice.dims}{self.timeslice.shape}: tile_range= {(tile_range[0], slice_end)}", display=True )
			return self.norm( batch, (tile_range[0],slice_end) )

	def norm(self, batch_data: xa.DataArray, tile_range: Tuple[int,int] ) -> xa.DataArray:
		channel_data = []
		ntype: str = cfg().task.norm
		channels: xa.DataArray = batch_data.coords['channels']
		for channel in channels.values:
			batch: xa.DataArray = batch_data.sel(channels=channel)
			if ntype == 'lnorm':
				bmean, bstd = batch.mean(dim=["x", "y"], skipna=True, keep_attrs=True), batch.std(dim=["x", "y"], skipna=True, keep_attrs=True)
				channel_data.append( (batch - bmean) / bstd )
			elif ntype == 'lscale':
				bmax, bmin = batch.max(dim=["x", "y"], skipna=True, keep_attrs=True), batch.min(dim=["x", "y"], skipna=True, keep_attrs=True)
				channel_data.append(  (batch - bmin) / (bmax-bmin) )
			elif ntype == 'gnorm':
				gstats: xa.DataArray = self.global_norm_stats.data_vars[channel]
				gmean, gstd = gstats.sel(stat='mean'), np.sqrt( gstats.sel(stat='var') )
				print( f"gnorm: gmean = {gmean.values}, gstd = {gstd.values}, batch mean = {batch.values.mean():.2f}, std = {batch.values.std():.2f}")
				channel_data.append(  (batch - gmean) / gstd )
			elif ntype == 'gscale':
				gstats: xa.DataArray = self.global_norm_stats.data_vars[channel]
				vmin, vmax = gstats.sel(stat='min'), gstats.sel(stat='max')
				channel_data.append(  (batch - vmin) / (vmax - vmin) )
			elif ntype == 'tnorm':
				tstats: xa.DataArray = self.norm_stats.data_vars[channel]
				tmean, tstd = tstats.sel(stat='mean').isel( tiles=slice(*tile_range) ), np.sqrt( tstats.sel(stat='var').isel( tiles=slice(*tile_range) ) )
				cbatch: np.ndarray = batch.values - tmean.values.reshape(-1,1,1)
				nbatch: np.ndarray = cbatch / tstd.values.reshape(-1,1,1)
				channel_data.append( batch.copy( data=nbatch) )
			elif ntype == 'tscale':
				tstats: xa.DataArray = self.norm_stats.data_vars[channel]
				vmin, vmax = tstats.sel(stat='min'), tstats.sel(stat='max')
				channel_data.append(  (batch - vmin) / (vmax - vmin) )
			else: raise Exception( f"Unknown norm: {ntype}")
		result = xa.concat( channel_data, channels ).transpose('tiles', 'channels', 'y', 'x')
		return result

	def get_tiles(self, var_data: List[np.ndarray]) -> xa.DataArray:
		raw_data = np.concatenate(var_data, axis=0)
		print( f"get_tiles: raw_data{raw_data.shape} mean = {np.nanmean(raw_data):.2f}, std = {np.nanstd(raw_data):.2f}")
		tsize: Dict[str, int] = self.tile_grid.get_full_tile_size()
		ishape = dict(c=raw_data.shape[0], y=raw_data.shape[1], x=raw_data.shape[2])
		grid_shape: Dict[str, int] = self.tile_grid.get_grid_shape( image_shape=ishape )
		roi: Dict[str, Tuple[int,int]] = self.tile_grid.get_active_region(image_shape=ishape)
		region_data: np.ndarray = raw_data[..., roi['y'][0]:roi['y'][1], roi['x'][0]:roi['x'][1]]
		lgm().log( f" ---- tsize{tsize}, grid_shape{grid_shape}, roi{roi}, ishape{ishape}, region_data{region_data.shape}",display=True)
		tile_data = region_data.reshape( ishape['c'], grid_shape['y'], tsize['y'], grid_shape['x'], tsize['x'] )
		tiles = np.swapaxes(tile_data, 2, 3).reshape( ishape['c'] * grid_shape['y'] * grid_shape['x'], tsize['y'], tsize['x'])
		msk = np.isfinite(tiles.mean(axis=-1).mean(axis=-1))
		ctiles = np.compress( msk, tiles,0)
		tile_idxs = np.compress(msk, np.arange(tiles.shape[0]), 0)
		result = ctiles.reshape( ctiles.shape[0]//ishape['c'], ishape['c'], tsize['y'], tsize['x'] )
		lgm().log(f" ---- tiles{tiles.shape}, tile_idxs{tile_idxs.shape}, ctiles{ctiles.shape} -> result{result.shape}",display=True)
		return xa.DataArray(result, dims=["tiles", "channels", "y", "x"], coords=dict(tiles=tile_idxs[0:result.shape[0]], channels=self.varnames) )