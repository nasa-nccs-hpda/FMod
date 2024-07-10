import xarray as xa, math, os
from fmod.base.util.config import cfg, dateindex
import pandas as pd
from datetime import  datetime, timedelta
from omegaconf import DictConfig, OmegaConf
from fmod.base.util.dates import drepr, date_list
from nvidia.dali import fn
from enum import Enum
from fmod.base.io.loader import TSet, srRes
from glob import glob
from typing import Any, Mapping, Sequence, Tuple, Union, List, Dict, Literal, Optional
from fmod.base.util.ops import format_timedeltas
from fmod.base.io.loader import data_suffix, path_suffix
from fmod.base.util.logging import lgm, exception_handled, log_timing
from fmod.base.io.loader import srRes, TSet
from fmod.base.source.loader.batch import SRDataLoader, FMDataLoader
import numpy as np

S = 'x'
CoordIdx = Union[ Dict[str,int], Tuple[int,int] ]

def cTup2Dict( c: CoordIdx ) -> Dict[str,int]:
	if type(c) is tuple: c = dict(x=c[0], y=c[1])
	return c

def dstr(date: datetime) -> str:
	return '{:04}{:02}{:02}{:02}'.format( date.year, date.month, date.day, date.hour )

def coords_filepath() -> str:
	return f"{cfg().dataset.dataset_root}/xy_coords.nc"

def i2x( c: str ) -> str:
	if c == "i": return "x"
	if c == "j": return "y"

def get_version(task_config: DictConfig) -> int:
	toks = task_config.dataset.split('-')
	tstr = "v0" if (len(toks) == 1) else toks[-1]
	assert tstr[0] == "v", f"Version str must start with 'v': '{tstr}'"
	return int(tstr[1:])

def datelist( date_range: Tuple[datetime, datetime] ) -> pd.DatetimeIndex:
	dlist =  pd.date_range( date_range[0], date_range[1], freq=f"{cfg().task.hours_per_step}h", inclusive="left" )
	# print( f" ^^^^^^ datelist[ {date_range[0].strftime('%H:%d/%m/%Y')} -> {date_range[1].strftime('%H:%d/%m/%Y')} ]: {dlist.size} dates, start_date = {cfg().task['start_date']}" )
	return dlist

def scale( varname: str, batch_data: np.ndarray ) -> np.ndarray:
	ranges: Dict[str,Dict[str,float]] = cfg().task.variable_ranges
	vrange: Dict[str,float] = ranges[varname]
	return (batch_data - vrange['min']) / (vrange['max'] - vrange['min'])

def tcoord( ** kwargs ) :
	dindx = kwargs.get('index',-1)
	date: Optional[datetime] = kwargs.get('date',None)
	return dindx if (date is None) else np.datetime64(date)

class S3ExportDataLoader(SRDataLoader):

	def __init__(self, task_config: DictConfig, tile_size: Dict[str, int], vres: srRes, tset: TSet,  **kwargs):
		SRDataLoader.__init__(self, task_config, vres)
		self.version = get_version( task_config )
		self.tset: TSet = tset
		self.coords_dataset: xa.Dataset = xa.open_dataset(coords_filepath(), **kwargs)
		#		self.xyc: Dict[str,xa.DataArray] = { c: self.coords_dataset.data_vars[ self.task.coords[c] ] for c in ['x','y'] }
		#		self.ijc: Dict[str,np.ndarray]   = { c: self.coords_dataset.coords['i'].values.astype(np.int64) for c in ['i','j'] }
		self.tile_size: Dict[str, int] = tile_size
		self.varnames: Dict[str, str] = self.task.input_variables
		self.use_memmap = task_config.get('use_memmap', False)
		self.shape = None

	def data_filepath(self, varname: str, **kwargs) -> Tuple[str,int]:
		root: str = cfg().dataset.dataset_root
		usf: int = math.prod(cfg().model.downscale_factors)
		dindx = kwargs.get('index',-2) + 1
		date: Optional[datetime] = kwargs.get('date',None)
		if date is not None:
			dindx = dateindex(date,self.task)
		self.dindxs.append(dindx)
		dset_params = dict( res=self.vres.value, index=f"{dindx:04}", varname=varname, tset=self.tset.value, usf=usf )
		for k,v in dset_params.items(): cfg().dataset[k] = v
		subpath: str = cfg().dataset.dataset_files[self.vres.value]
		fpath = f"{root}/{subpath}"
		return fpath, dindx

	def dataset_glob(self, varname: str) -> str:
		root: str = cfg().dataset.dataset_root
		usf: int = math.prod(cfg().model.downscale_factors)
		dset_params = dict( res=self.vres.value, varname=varname, index=f"*", tset=self.tset.value, usf=usf )
		for k,v in dset_params.items(): cfg().dataset[k] = v
		subpath: str = cfg().dataset.dataset_files[self.vres.value]
		fglob = f"{root}/{subpath}"
		return fglob

	def get_dset_size(self) -> int:
		varname: Tuple[str,str] = list(self.varnames.items())[0]
		dsglob = self.dataset_glob( varname[0] )
		dss = len( glob(dsglob) )
		print( f" ************ get_dset_size: glob='{dsglob}', size={dss}")
		return dss

	# def cut_coord(self, oindx: Dict[str,int], c: str) -> np.ndarray:
	# 	cdata: np.ndarray = self.ijc[c]
	# 	return cdata[origin[i2x(c)]: origin[i2x(c)] + self.tile_size[i2x(c)] ]

	def cut_tile( self, idx: int, data_grid: np.ndarray, origin: Dict[str,int] ) -> np.ndarray:
		tile_bnds = [ origin['y'], origin['y'] + self.tile_size['y'], origin['x'], origin['x'] + self.tile_size['x'] ]
		result: np.ndarray = data_grid[ tile_bnds[0]: tile_bnds[1], tile_bnds[2]: tile_bnds[3] ]
		lgm().debug( f"     ------------------>> cut_tile[{idx}]: origin={list(origin.values())}, tile_bnds = {tile_bnds}" )
		return result

	def cut_domain( self, timeslice_data: np.ndarray ):
		origin: Dict[str,int] = cfg().task.origin[self.tset.value]
		tile_grid: Dict[str,int] = cfg().task.tile_grid[self.tset.value]
		tile_bnds = { c:  [origin[c], origin[c]+self.tile_size[c]*tile_grid[c]] for c in ['x','y'] }
		lgm().debug( f"     ------------------>> cut_domain: origin={origin}, tile_bnds = {tile_bnds}")
		return timeslice_data[ tile_bnds['y'][0]:tile_bnds['y'][1], tile_bnds['x'][0]:tile_bnds['x'][1] ]

	# def cut_xy_coords(self, oindx: Dict[str,int] )-> Dict[str,xa.DataArray]:
	# 	tcoords: Dict[str,np.ndarray] = { c:  self.cut_coord( origin, c ) for idx, c in enumerate(['i','j']) }
	# #	xycoords: Dict[str,xa.DataArray] = { cv: xa.DataArray( self.cut_tile( self.xyc[cv].values, origin ), dims=['j','i'], coords=tcoords ) for cv in ['x','y'] }
	# #	xycoords: Dict[str, xa.DataArray] = {cv[0]: xa.DataArray(tcoords[cv[1]].astype(np.float32), dims=[cv[1]], coords=tcoords) for cv in [('x','i'), ('y','j')]}
	# 	xc = xa.DataArray(tcoords['i'].astype(np.float32), dims=['i'], coords=dict(i=tcoords['i']))
	# 	yc = xa.DataArray(tcoords['j'].astype(np.float32), dims=['j'], coords=dict(j=tcoords['j']))
	# 	return dict(x=xc, y=yc) #, **tcoords)

	def open_timeslice(self, vid: str, **kwargs) -> np.memmap:
		fpath, fidex = self.data_filepath( vid, **kwargs )
		mmap_mode = 'r' if self.use_memmap else None
		raw_data: np.memmap = np.load(fpath, allow_pickle=True, mmap_mode=mmap_mode)
		if self.shape is None:
			self.shape = list(raw_data.shape)
			lgm().log( f"Loaded {vid}({fidex}): shape={self.shape}", display=True )
		return raw_data


	def load_global_timeslice(self, vid: str, **kwargs) -> np.ndarray:
		fpath, fidex = self.data_filepath( vid, **kwargs )
		mmap_mode = 'r' if self.use_memmap else None
		timeslice: np.memmap = np.load(fpath, allow_pickle=True, mmap_mode=mmap_mode)
		return self.cut_domain(timeslice)

	def load_channel( self, idx: int, origin: CoordIdx, vid: Tuple[str,str], **kwargs ) -> xa.DataArray:
		raw_data: np.memmap = self.open_timeslice(vid[0], **kwargs)
		tile_data: np.ndarray = self.cut_tile( idx, raw_data, cTup2Dict(origin) )
		result = xa.DataArray( scale( vid[0], tile_data ), dims=['y', 'x'],  attrs=dict( fullname=vid[1] ) ) # coords=dict(**tc, **tc['x'].coords, **tc['y'].coords),
		result = result.expand_dims( axis=0, dim=dict(channels=[vid[0]]) )
		# print( f"load_channel: shape = {result.shape}, raw_data shape = {raw_data.shape}, tile_data shape = {tile_data.shape}")
		return result

	def load_timeslice( self, idx: int, origin: CoordIdx, **kwargs ) -> xa.DataArray:
		arrays: List[xa.DataArray] = [ self.load_channel( idx, origin, vid, **kwargs ) for vid in self.varnames.items() ]
		result = xa.concat( arrays, "channels" )
		result = result.expand_dims(axis=0, dim=dict(time=[tcoord(**kwargs)]))
		return result

	def load_temporal_batch( self, origin: CoordIdx, date_range: Tuple[datetime,datetime] ) -> xa.DataArray:
		timeslices = [ self.load_timeslice( idx, origin, date=date ) for idx, date in enumerate( datelist( date_range ) ) ]
		result = xa.concat(timeslices, "time")
		lgm().log( f" ** load-batch-{self.vres.value} [{date_range[0]}]:{result.dims}:{result.shape}, origin={origin}, tilesize = {self.tile_size}" )
		return result

	def load_index_batch( self, origin: CoordIdx, index_range: Tuple[int,int] ) -> xa.DataArray:
		timeslices = [ self.load_timeslice( idx, origin, index=idx ) for idx in range( *index_range ) ]
		result = xa.concat(timeslices, "time")
		lgm().log( f" ** load-batch-{self.vres.value} [{index_range[0]}]:{result.dims}:{result.shape}, origin={origin}, tilesize = {self.tile_size}" )
		return result

	def load_norm_data(self) -> Dict[str,xa.DataArray]:
		return {}

	def load_const_dataset(self, origin: CoordIdx )-> Optional[xa.DataArray]:
		return None





